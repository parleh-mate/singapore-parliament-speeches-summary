import pandas as pd
import json
from google.cloud import storage
import pickle

from params.table_ids import speeches_table_id, positions_table_id, bills_table_id, bill_summaries_table_id, bill_split_summaries_table_id

def get_last_job(gbq_client, dim_table_id):

    job = gbq_client.query(f"""
                        SELECT *
                        FROM `{dim_table_id}`
                            order by batch_id desc
                            limit 1
                        """)

    result = job.result()
    batch_check_df = result.to_dataframe()

    return batch_check_df

def get_last_job_meta(gbq_client, dim_table_id):
    
    batch_check_df = get_last_job(gbq_client, dim_table_id)
    # get batch ids
    gpt_batch_id = batch_check_df.gpt_batch_id[0]
    batch_id = batch_check_df.batch_id[0]
    # get status
    batch_status = batch_check_df.status.iloc[0]
    return {"gpt_batch_id": gpt_batch_id, "batch_id": batch_id, "batch_status": batch_status}


def retrieve_batch_meta(gpt_batch_id, gpt_client):
    
    batch_meta = gpt_client.batches.retrieve(gpt_batch_id)

    return batch_meta


def extract_policy_positions(batch_meta, gpt_client, batch_id, gpt_batch_id, topics_list):

    file_response = gpt_client.files.content(batch_meta.output_file_id)
    text_response = file_response.read().decode('utf-8')
    text_dict = [json.loads(obj) for obj in text_response.splitlines()]

    data = []
    for i in text_dict:
        try:
            speech_id = i['custom_id']
            output = eval(i['response']['body']['choices'][0]['message']['content'])
            policy_positions = output['Positions']
            topic_assigned = output['Topic']
            # only append if the topic exists
            if topic_assigned in topics_list:
                data.append((speech_id, policy_positions, topic_assigned))
        except:
            pass

    speech_positions = pd.DataFrame(data, columns=['speech_id', 'policy_positions', 'topic_assigned'])
    speech_positions['batch_id'] = batch_id
    speech_positions['gpt_batch_id'] = gpt_batch_id

    return speech_positions

def extract_bill_summaries(batch_meta, gpt_client, batch_id, gpt_batch_id):

    file_response = gpt_client.files.content(batch_meta.output_file_id)
    text_response = file_response.read().decode('utf-8')
    text_dict = [json.loads(obj) for obj in text_response.splitlines()]

    data = []
    for i in text_dict:
        try:
            bill_number = i['custom_id']
            output = eval(i['response']['body']['choices'][0]['message']['content'])
            bill_introduction = output['bill_introduction']
            bill_key_points = output['bill_key_points']
            bill_impact = output['bill_impact']
            data.append((bill_number, bill_introduction, bill_key_points, bill_impact))
        except Exception as e:
            print(f"Caught an exception while extracting bill summaries from GPT: {e}")
            print(f"Error type: {type(e).__name__}")
            pass

    bill_summaries = pd.DataFrame(data, columns=['bill_number', 'bill_introduction', 'bill_key_points', 'bill_impact'])
    bill_summaries['batch_id'] = batch_id
    bill_summaries['gpt_batch_id'] = gpt_batch_id

    return bill_summaries

def extract_long_bill_summaries(batch_meta, gpt_client, batch_id, gpt_batch_id):

    file_response = gpt_client.files.content(batch_meta.output_file_id)
    text_response = file_response.read().decode('utf-8')
    text_dict = [json.loads(obj) for obj in text_response.splitlines()]

    data = []
    for i in text_dict:
        bill_number = i['custom_id']
        output = eval(i['response']['body']['choices'][0]['message']['content'])
        bill_split_key_points = output['bill_split_key_points']
        data.append((bill_number, bill_split_key_points))

    bill_split_summaries = pd.DataFrame(data, columns=['bill_number', 'bill_split_key_points'])
    bill_split_summaries['batch_id'] = batch_id
    bill_split_summaries['gpt_batch_id'] = gpt_batch_id

    bill_split_summaries[['bill_number', 'bill_split']] = bill_split_summaries['bill_number'].str.split('-', expand=True)
    bill_split_summaries['bill_split'] = bill_split_summaries['bill_split'].astype(int)

    return bill_split_summaries

def get_failed_long_bills(gbq_client):

    job = gbq_client.query(f"""
                        SELECT *, 'failed' as status
                        FROM `{bill_split_summaries_table_id}`
                        where bill_number not in (select bill_number from {bill_summaries_table_id})
                        """)

    result = job.result()
    failed_long_bills = result.to_dataframe()

    return failed_long_bills

def get_unsummarized_speeches(gbq_client, upper_bound, lower_bound, batch_size):
    job = gbq_client.query(f"""
                        SELECT distinct *
                        FROM `{speeches_table_id}`
                        WHERE speech_id not in (select speech_id from `{positions_table_id}`)
                        AND topic_type_name not like "%Correction by Written Statements%"
                        AND not is_primary_question
                        AND topic_type_name not like "%Bill Introduced%"
                        AND count_speeches_words<{upper_bound} 
                        AND count_speeches_words>{lower_bound}
                        AND member_name != ''
                        AND member_constituency is not NULL
                        LIMIT {batch_size}
                        """)

    result = job.result()
    df = result.to_dataframe()

    return df

def get_unsummarized_bills(gbq_client, batch_size):
    job = gbq_client.query(f"""
                       SELECT * 
                       FROM `{bills_table_id}`
                       WHERE pdf_link is not NULL
                       AND number not in (select bill_number from `{bill_summaries_table_id}`)
                       AND number not in (select distinct bill_number from `{bill_split_summaries_table_id}`)
                       LIMIT {batch_size}
                       """)
    result = job.result()
    df = result.to_dataframe()

    return df

def get_passed_bills(gbq_client):
    job = gbq_client.query(f"""
    select *
    from `{bills_table_id}`
    where date_passed is not NULL
    """)
    result = job.result()
    return result.to_dataframe()

def get_dash_cached_datasets():
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket("dash-app-cache")
    blob = bucket.blob("dash-datasets")

    # Download the serialized data
    serialized_data = blob.download_as_bytes()

    # Deserialize the data back into a dictionary
    return pickle.loads(serialized_data)

def collect_speech_meta(gbq_client, speech_ids):
    job = gbq_client.query(f"""
                    SELECT speech_id, `date`, parliament, member_name, member_party, member_constituency
                    FROM `{speeches_table_id}`
                    WHERE speech_id IN ({','.join([f"'{i}'" for i in speech_ids])}
                    )
                    """)

    result = job.result()
    metadata_df = result.to_dataframe()
    return metadata_df

def collect_bill_meta(gbq_client, bill_numbers):
    job = gbq_client.query(f"""
                    SELECT number as bill_number, title, pdf_link, date(date_introduced) as date_introduced, date(date_passed) as date_passed, parliament
                    FROM `{bills_table_id}` bills
                    left join `singapore-parliament-speeches.prod_stg.stg_gsheet_parliament_dates` dates
                    on date(bills.date_introduced) BETWEEN dates.from_date and dates.to_date
                    WHERE number IN ({','.join([f"'{i}'" for i in bill_numbers])}
                    )
                    """)

    result = job.result()
    metadata_df = result.to_dataframe()
    return metadata_df