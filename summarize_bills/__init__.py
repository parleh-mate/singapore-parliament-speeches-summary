import pandas as pd
from datetime import date
import time
import sys


from extract import extract_bill_summaries, extract_long_bill_summaries, get_unsummarized_bills, retrieve_batch_meta, collect_bill_meta, get_last_job_meta, get_failed_long_bills, get_passed_bills, get_dash_cached_datasets

from load import create_or_append_table, upload_embedding_meta_gbq

from utils import change_to_completed, create_embeddings_job, upload_embeddings_batch_job, upsert_pinecone, extract_finished_embeddings, prepare_bill_data_upsert, delete_in_progress, create_json_batch_file, upload_batch_to_gpt, download_and_extract_bill_pdf, count_bill_tokens, split_bills_df_to_parts

from params import bills_batch_size, bills_model, local_bills_split_batch_directory, local_bills_batch_directory, gbq_bill_summaries_schema, gbq_bill_split_summaries_schema, bill_embeddings_job_description

from params.gpt_prompts import bills_split_summary_response_format, bills_split_summary_system_message, output_bill_introduction_description, output_bill_key_points_description, output_bill_impact_description, bills_summary_system_message, bills_summary_response_format, bills_split_key_points_description

from params.table_ids import dim_bill_summaries_table_id, dim_bill_split_summaries_table_id, bill_summaries_table_id, dim_bill_embeddings_table_id, bill_split_summaries_table_id, bills_table_id

def handle_finished_bills(batch_meta, batch_id, gpt_batch_id, gpt_client, gbq_client):
    bill_summaries = extract_bill_summaries(batch_meta, gpt_client, batch_id, gpt_batch_id) 
    # upload to GBQ
    create_or_append_table(bill_summaries, bill_summaries_table_id, gbq_client)
    # update status
    change_to_completed(gbq_client, dim_bill_summaries_table_id)

    return bill_summaries

def handle_finished_long_bills(batch_meta, batch_id, gpt_batch_id, gpt_client, gbq_client):
    bill_split_summaries = extract_long_bill_summaries(batch_meta, gpt_client, batch_id, gpt_batch_id) 
    # upload to GBQ
    create_or_append_table(bill_split_summaries, bill_split_summaries_table_id, gbq_client)
    # update status
    change_to_completed(gbq_client, dim_bill_split_summaries_table_id)

    return bill_split_summaries

def extract_finished_long_bills(gpt_client, gbq_client):
    # collect long bills that are split and summarized but do not appear in final summaries (job failed, etc)
    long_bills_failed = get_failed_long_bills(gbq_client)

    # get long bills that are not yet extracted from batch job
    long_bills_meta =  get_last_job_meta(gbq_client, dim_bill_split_summaries_table_id)
    bill_split_summaries = None # initial value

    # only continue if a job was scheduled
    if (long_bills_meta['batch_status']=='in_progress'):
        batch_meta = retrieve_batch_meta(long_bills_meta['gpt_batch_id'], gpt_client)
        # only extract batch data if the job is completed
        if batch_meta.status=="completed":
            bill_split_summaries = handle_finished_long_bills(batch_meta, long_bills_meta['batch_id'], long_bills_meta['gpt_batch_id'], gpt_client, gbq_client)
            bill_split_summaries['status'] = 'new'

        elif batch_meta.status=="failed":
            # if batch fails, delete in GBQ
            delete_in_progress(gbq_client, dim_bill_split_summaries_table_id)
        else:
            print("bill split job still in progress")

    # merge all parts into one whole
    bill_split_summaries = pd.concat([bill_split_summaries, long_bills_failed])
    bill_split_summaries['bill_split_key_points'] = bill_split_summaries.groupby('bill_number')['bill_split_key_points'].transform(lambda x: ''.join(x))
    bill_split_summaries = bill_split_summaries.groupby('bill_number').head(1).drop(columns = ['bill_split'])
    return bill_split_summaries

def handle_bill_summaries_embeddings(bill_summaries, gbq_client, gpt_client, index):
    # now create embeddings job
    bill_summaries_combined = bill_summaries.apply(lambda x: "introduction: " + x.bill_introduction + "key_points: " + x.bill_key_points + "impact: " + x.bill_impact, axis = 1)
    bill_summaries_combined.name = 'bill_summary'
    bill_numbers = bill_summaries.bill_number

    create_embeddings_job(bill_summaries_combined, bill_numbers)
    batch_meta = upload_embeddings_batch_job(gpt_client, bill_embeddings_job_description)
    
    # now pause 120 seconds and check if batch job finished, if not, pause again 120 seconds
    while (batch_meta.status!='completed'):
        batch_meta = retrieve_batch_meta(batch_meta.id, gpt_client)
        time.sleep(120)
        if (batch_meta.status=='failed'):
            sys.exit()

    # once job is finished, upsert to pinecone
    embed_dict = extract_finished_embeddings(gpt_client, batch_meta)
    bill_meta_df = collect_bill_meta(gbq_client, bill_numbers)
    vectors_list = prepare_bill_data_upsert(bill_meta_df, bill_summaries, embed_dict)
    # Upsert to pinecone in batches of 1
    upsert_pinecone(vectors_list, index, 1)
    upload_embedding_meta_gbq(gbq_client, dim_bill_embeddings_table_id, batch_meta.id, "bill_number", bill_numbers)

def handle_bill_summaries_extraction(gbq_client, gpt_client, index):
    last_job_meta = get_last_job_meta(gbq_client, dim_bill_summaries_table_id)
    # only continue if a job was scheduled
    if (last_job_meta['batch_status']=='in_progress'):
        batch_meta = retrieve_batch_meta(last_job_meta['gpt_batch_id'], gpt_client)
        # only extract batch data if the job is completed
        if batch_meta.status=="completed":
            print("extracting completed bill summaries")
            bill_summaries = extract_bill_summaries(batch_meta, gpt_client, last_job_meta['batch_id'], last_job_meta['gpt_batch_id']) 

            if len(bill_summaries!=0):
                print("embedding bill summaries")
                handle_bill_summaries_embeddings(bill_summaries, gbq_client, gpt_client, index)
                # only upload to gbq and mark job as completed if embeddings successful, otherwise script exits
                # upload to GBQ
                create_or_append_table(bill_summaries, bill_summaries_table_id, gbq_client)
                # update status
                change_to_completed(gbq_client, dim_bill_summaries_table_id)
                print(f"{len(bill_summaries)} bill summaries embedded from scheduled batch size of {batch_meta.request_counts.completed}")
            else:
                print(f"No bill summaries to embed from scheduled batch size of {batch_meta.request_counts.completed}")
        elif batch_meta.status=="failed":
            # if batch fails, delete in GBQ
            delete_in_progress(gbq_client, dim_bill_summaries_table_id)
        else:
            print("job still in progress")
    else:
        print("no jobs scheduled, proceed to create job")

def create_long_bills_job(df, gbq_client, gpt_client):
    last_job_meta =  get_last_job_meta(gbq_client, dim_bill_split_summaries_table_id)
     # only continue if last job was completed
    if last_job_meta['batch_status']=='completed':   
        bill_numbers = df.number.tolist()
        bill_splits = df.bill_splits.tolist()
        df = split_bills_df_to_parts(df)
        df['number'] = df['number'] + '-' + df['part'].astype(str)

        # send split df to gpt
        create_json_batch_file(df, 'split_text', 'number', bills_model, bills_split_summary_system_message, bills_split_summary_response_format, local_bills_split_batch_directory)

        split_bills_gpt_batch_id = upload_batch_to_gpt(gpt_client,
                                                    local_bills_split_batch_directory,
                                                    "singapore parliamentary split bills summary batch job")
        
        # new batch id and date
        split_bills_batch_id = last_job_meta['batch_id'] + 1
        batch_date = date.today()

        # ready new batch meta to be uploaded to gbq
        dim_bill_split_summaries = pd.DataFrame({"batch_id": [split_bills_batch_id],
                                                "gpt_batch_id": [split_bills_gpt_batch_id],
                                                "model": [bills_model],
                                                "batch_date": [batch_date],
                                                "system_message": [bills_split_summary_system_message],
                                                "bills_split_key_points_description": [bills_split_key_points_description],
                                                "bill_numbers":[bill_numbers],
                                                "bill_splits":[bill_splits],      
                                                "status": ["in_progress"]})

        # upload batch meta to gbq
        create_or_append_table(dim_bill_split_summaries, dim_bill_split_summaries_table_id, gbq_client, gbq_bill_split_summaries_schema)
    else:
        print("Previous long bills job not yet completed, current long bills will not be added")

def handle_bill_summaries_creation(gbq_client, gpt_client):
    last_job_meta = get_last_job_meta(gbq_client, dim_bill_summaries_table_id)
    # prevent starting if previous job not finished
    if (last_job_meta['batch_status']=='completed'):

        # get newly completed or previously failed split long bills
        bill_split_summaries = extract_finished_long_bills(gpt_client, gbq_client)
        new_bills_batch_size = bills_batch_size - len(bill_split_summaries)

        unsummarized_rest_df = None # default value
        # create new bill summaries job
        if new_bills_batch_size>0:
            unsummarized_df = get_unsummarized_bills(gbq_client, new_bills_batch_size)
            if len(unsummarized_df)==0:
                print("No new bills to summarize, exiting script.")
                return
            unsummarized_df['bill_text'] = unsummarized_df['pdf_link'].transform(lambda x: download_and_extract_bill_pdf(x))
            unsummarized_df['num_tokens'] = unsummarized_df['bill_text'].transform(lambda x: count_bill_tokens(x))
            unsummarized_df['bill_splits'] = unsummarized_df['num_tokens'].transform(lambda x: -(x//-(128000*0.9))) # ceil division, add 10% buffer

            # deal with speeches that need to be split first, split into equal increments on string length
            unsummarized_split_df = unsummarized_df.query("bill_splits>1")
            if len(unsummarized_split_df)>0:
                create_long_bills_job(unsummarized_split_df, gbq_client, gpt_client)        

            # gather rest unsplit
            unsummarized_rest_df = unsummarized_df.query("bill_splits==1")

        unsummarized_df = pd.concat([bill_split_summaries.rename(columns = {'bill_number': 'number', 'bill_split_key_points': 'bill_text'})[['bill_text', 'number']],
                                     unsummarized_rest_df[['bill_text', 'number']]])

        create_json_batch_file(unsummarized_df, 'bill_text', 'number', bills_model,bills_summary_system_message, bills_summary_response_format, local_bills_batch_directory)

        # now upload this batch to gpt and get the batch_id
        bills_gpt_batch_id = upload_batch_to_gpt(gpt_client,
                                                 local_bills_batch_directory, 
                                                 "singapore parliamentary bills summary batch job")

        # new batch id and date
        bills_batch_id = last_job_meta['batch_id'] + 1
        batch_date = date.today()

        # ready new batch meta to be uploaded to gbq
        dim_bill_summaries = pd.DataFrame({"batch_id": [bills_batch_id],
                                           "gpt_batch_id": [bills_gpt_batch_id],
                                           "model": [bills_model],
                                           "batch_date": [batch_date],
                                           "system_message": [bills_summary_system_message],
                                           "output_bill_introduction_description":
                                           [output_bill_introduction_description],
                                           "output_bill_key_points_description": [output_bill_key_points_description],
                                           "output_bill_impact_description": [output_bill_impact_description],            
                                           "status": ["in_progress"]})

        # upload batch meta to gbq
        create_or_append_table(dim_bill_summaries, dim_bill_summaries_table_id, gbq_client, gbq_bill_summaries_schema)
    
    else:
        print("Tried to create jobs when previous one not completed.")     

def check_bill_updates(gbq_client, zilliz_client):
    print("checking for bill updates")
    index_df = get_dash_cached_datasets()['bill_summaries']

    # check which bills not yet passed in df
    unpassed_bills = index_df.query('date_passed=="NaT"')["bill_number"].to_list()
    print(f"The following bills are unpassed in the dataset: {', '.join(unpassed_bills)}")
    # check whether they've actually been passed
    passed_bills = get_passed_bills(gbq_client)

    # check which bills needs to be updated
    bills_to_update = passed_bills[passed_bills.number.isin(unpassed_bills)]
    print(f"found {len(bills_to_update)} bills to update")
    if len(bills_to_update)>0:
        # update zilliz index
        bills_to_update["date_passed"] = bills_to_update["date_passed"].dt.strftime("%Y-%m-%d")

        for ind, row in bills_to_update.iterrows():
            rag_entry = zilliz_client.query(collection_name="singapore_bill_summaries", filter=f'id == "{row.number}"', output_fields=["id", "vector", "namespace", "bill_impact", "bill_introduction", "bill_key_points", "date_introduced", "date_passed", "parliament", "pdf_link", "title"])[0]

            rag_entry["date_passed"] = row.date_passed

            zilliz_client.upsert(collection_name="singapore_bill_summaries",data=rag_entry)

        print(f"{len(bills_to_update)} bills updated.")
    else:
        print("No bills updated.")
    