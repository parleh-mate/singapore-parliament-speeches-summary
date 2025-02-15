import pandas as pd
from datetime import date
import time
import sys

from extract import extract_policy_positions, retrieve_batch_meta, collect_speech_meta, get_last_job_meta, get_unsummarized_speeches

from load import create_or_append_table, upload_embedding_meta_gbq

from utils import change_to_completed, create_embeddings_job, upload_embeddings_batch_job, upsert_pinecone, extract_finished_embeddings, prepare_speech_data_upsert, delete_in_progress, create_json_batch_file, upload_batch_to_gpt

from params import topics_list, speech_embeddings_job_description, word_upper_bound, word_lower_bound, speech_batch_size, positions_model, local_positions_batch_directory, gbq_positions_schema

from params.table_ids import dim_positions_table_id, dim_positions_embeddings_table_id, positions_table_id

from params.gpt_prompts import positions_summary_system_message, output_position_description, output_topic_description, positions_summary_response_format

def handle_finished_positions(batch_meta, batch_id, gpt_batch_id, gpt_client, gbq_client):
    speech_positions = extract_policy_positions(batch_meta, gpt_client, batch_id, gpt_batch_id, topics_list)
    # now upload policy positions to GBQ
    create_or_append_table(speech_positions, positions_table_id, gbq_client)
    # update status
    change_to_completed(gbq_client, dim_positions_table_id)
    return speech_positions

def handle_positions_embeddings(speech_positions, gbq_client, gpt_client, index):
    # now create embeddings job
    positions = speech_positions.policy_positions
    speech_ids = speech_positions.speech_id

    create_embeddings_job(positions, speech_ids)
    batch_meta = upload_embeddings_batch_job(gpt_client, speech_embeddings_job_description)
    
    # now pause 120 seconds and check if batch job finished, if not, pause again 120 seconds
    while (batch_meta.status!='completed'):
        batch_meta = retrieve_batch_meta(batch_meta.id, gpt_client)
        time.sleep(120)
        if (batch_meta.status=='failed'):
            sys.exit()

    # once job is finished, upsert to pinecone
    embed_dict = extract_finished_embeddings(gpt_client, batch_meta)
    speech_meta_df = collect_speech_meta(gbq_client, speech_ids)
    vectors_list = prepare_speech_data_upsert(speech_meta_df, speech_positions, embed_dict)
    # Upsert to pinecone in batches of 100
    upsert_pinecone(vectors_list, index, 100)
    upload_embedding_meta_gbq(gbq_client, dim_positions_embeddings_table_id, batch_meta.id, "speech_ids", speech_ids)

def handle_positions_extraction(gbq_client, gpt_client, index):
    last_job_meta = get_last_job_meta(gbq_client, dim_positions_table_id)
    # only continue if a job was scheduled
    if (last_job_meta['batch_status']=='in_progress'):
        batch_meta = retrieve_batch_meta(last_job_meta['gpt_batch_id'], gpt_client)
        # only extract batch data if the job is completed
        if batch_meta.status=="completed":
            speech_positions = handle_finished_positions(batch_meta, last_job_meta['batch_id'], last_job_meta['gpt_batch_id'], gpt_client, gbq_client)
            if len(speech_positions)!=0:
                handle_positions_embeddings(speech_positions, gbq_client, gpt_client, index)
                print(f"{len(speech_positions)} speech positions embedded from scheduled batch size of {batch_meta.request_counts.completed}")
            else:
                print(f"No speech positions to embed from scheduled batch size of {batch_meta.request_counts.completed}")
        elif batch_meta.status=="failed":
            # if batch fails, delete in GBQ
            delete_in_progress(gbq_client, dim_positions_table_id)
        else:
            print("job still in progress")
    else:
        print("no jobs scheduled, proceed to create job")

def handle_positions_creation(gbq_client, gpt_client):
    last_job_meta = get_last_job_meta(gbq_client, dim_positions_table_id)
    # prevent starting if previous job not finished
    if (last_job_meta['batch_status']=='completed'):

        # now create new speech positions job
        unsummarized_df = get_unsummarized_speeches(gbq_client, word_upper_bound, word_lower_bound, speech_batch_size)
        create_json_batch_file(unsummarized_df, "speech_text", "speech_id", positions_model, positions_summary_system_message, positions_summary_response_format, local_positions_batch_directory)

        # now upload this batch to gpt and get the batch_id
        positions_gpt_batch_id = upload_batch_to_gpt(gpt_client, local_positions_batch_directory, speech_embeddings_job_description)

        # new batch id and date
        positions_batch_id = last_job_meta['batch_id'] + 1
        batch_date = date.today()

        # ready new batch meta to be uploaded to gbq
        dim_speech_positions = pd.DataFrame({"batch_id": [positions_batch_id],
                                            "gpt_batch_id": [positions_gpt_batch_id],
                                            "model": [positions_model],
                                            "batch_date": [batch_date],
                                            "system_message": [positions_summary_system_message],
                                            "output_position_description": [output_position_description],
                                            "output_topic_description": [output_topic_description],
                                            "lower_word_bound": [word_lower_bound],
                                            "upper_word_bound": [word_upper_bound],
                                            "status": ["in_progress"]})

        # upload batch meta to gbq
        create_or_append_table(dim_speech_positions, dim_positions_table_id, gbq_client, gbq_positions_schema)
    
    else:
        print("Tried to create jobs when previous one not completed.")