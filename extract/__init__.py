import pandas as pd
import json

def get_in_progress_jobs(gbq_client, dim_table_id):

    job = gbq_client.query(f"""
                        SELECT *
                        FROM `{dim_table_id}`
                            WHERE status = 'in_progress'
                        """)

    result = job.result()
    batch_check_df = result.to_dataframe()

    return batch_check_df


def retrieve_batch_meta(gpt_batch_id, gpt_client):
    
    batch_meta = gpt_client.batches.retrieve(gpt_batch_id)

    return batch_meta


def extract_summarized_data(batch_meta, gpt_client, batch_id, gpt_batch_id, topics_list):

    file_response = gpt_client.files.content(batch_meta.output_file_id)
    text_response = file_response.read().decode('utf-8')
    text_dict = [json.loads(obj) for obj in text_response.splitlines()]

    data = []
    for i in text_dict:
        try:
            speech_id = i['custom_id']
            output = eval(i['response']['body']['choices'][0]['message']['content'])
            speech_summary = output['Summary']
            topic_assigned = output['Topic']
            # only append if the topic exists
            if topic_assigned in topics_list:
                data.append((speech_id, speech_summary, topic_assigned))
        except:
            pass

    speech_summaries = pd.DataFrame(data, columns=['speech_id', 'speech_summary', 'topic_assigned'])
    speech_summaries['batch_id'] = batch_id
    speech_summaries['gpt_batch_id'] = gpt_batch_id

    return speech_summaries

def get_unsummarized_speeches(gbq_client, upper_bound, lower_bound, batch_size):
    job = gbq_client.query(f"""
                        SELECT  *
                        FROM `singapore-parliament-speeches.prod_mart.mart_speeches`
                        WHERE speech_id not in (select speech_id from `singapore-parliament-speeches.prod_mart.mart_speech_summaries`)
                        AND topic_type_name not like "%Correction by Written Statements%"
                        AND topic_type_name not like "%Bill Introduced%"
                        AND count_speeches_words<{upper_bound} 
                        AND count_speeches_words>{lower_bound}
                        AND member_name != ''
                        AND member_name != 'Speaker'
                        LIMIT {batch_size}
                        """)

    result = job.result()
    df = result.to_dataframe()

    return df