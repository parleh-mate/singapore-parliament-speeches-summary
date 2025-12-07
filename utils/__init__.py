import jsonlines
import json
import pandas as pd
import requests
import pymupdf4llm
import tiktoken

from params import embedding_model, embedding_dimensions, bills_model, bill_pdf_directory
from params.table_ids import dim_bill_split_summaries_table_id, bill_summaries_table_id

def change_to_completed(gbq_client, dim_table_id):
    job = gbq_client.query(f"""
                            UPDATE `{dim_table_id}`
                            SET status = 'completed'
                            WHERE status = 'in_progress'
                            """)
    job.result()

def delete_in_progress(gbq_client, dim_table_id):
    job = gbq_client.query(f"""
                            delete from `{dim_table_id}`
                            WHERE status = 'in_progress'
                            """)
    job.result()

def create_json_batch_file(df, input_text_var, id_var, model, system_message, response_format, local_batch_directory):
    json_list = []
    for ind,row in df.iterrows():

        input = row[input_text_var]   

        json_list.append({"custom_id": row[id_var], 
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, 
                "messages": [{"role": "system", 
                                "content": system_message},
                                {"role": "user", 
                                "content": input}],
                                "max_tokens": 3000,
                                "response_format": response_format
        }})

    with jsonlines.open(local_batch_directory, 'w') as writer:
        writer.write_all(json_list)


def upload_batch_to_gpt(gpt_client, local_batch_directory, description):
    batch_input_file = gpt_client.files.create(
    file=open(local_batch_directory, "rb"),
    purpose="batch"
    )

    batch_file_id = batch_input_file.id

    batch_meta = gpt_client.batches.create(
        input_file_id=batch_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": description
        }
    )

    return batch_meta.id

def create_embeddings_job(content, ids):
    json_list = []
    for i in range(len(content)): 

        json_list.append({"custom_id": ids[i], 
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"input": content[i],
                "model": embedding_model,
                "encoding_format": "float",
                'dimensions': embedding_dimensions
                }})

    with jsonlines.open("assets/batch_embeddings.jsonl", 'w') as writer:
        writer.write_all(json_list)

def upload_embeddings_batch_job(gpt_client, description):
    batch_input_file = gpt_client.files.create(
        file=open("assets/batch_embeddings.jsonl", "rb"),
        purpose="batch"
        )

    batch_file_id = batch_input_file.id

    batch_meta = gpt_client.batches.create(
        input_file_id=batch_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
        "description": description
        }
    )
    return batch_meta

def extract_finished_embeddings(gpt_client, batch_meta):
    file_response = gpt_client.files.content(batch_meta.output_file_id)
    embed_response = file_response.read().decode('utf-8')
    embed_dict = [json.loads(obj) for obj in embed_response.splitlines()]
    return embed_dict

def prepare_speech_df_upsert(speech_meta_df, content):
    upsert_df = pd.merge(speech_meta_df, 
            content.filter(['speech_id', 'policy_positions', 'topic_assigned']), 
            'left', 
            'speech_id')

    # convert columns to accepted formats for pinecone
    upsert_df['date'] = upsert_df.date.astype(str)
    upsert_df['parliament'] = upsert_df.parliament.astype(object)
    return upsert_df

def prepare_speech_data_upsert(speech_meta_df, positions, embed_dict):
    upsert_df = prepare_speech_df_upsert(speech_meta_df, positions)
    vectors_list = []
    for i in embed_dict:
        # get speech_id
        speech_id = i['custom_id']
        
        # get embeddings
        embeddings = list(i['response']['body']['data'][0]['embedding'])

        # get speech_summary and metadata
        meta_df = upsert_df.query(f"speech_id=='{speech_id}'")
        
        vector = {"id": speech_id, 
                  "vector": embeddings,
                  "date": meta_df['date'].iloc[0],
                  "parliament": meta_df['parliament'].iloc[0],
                  "name": meta_df['member_name'].iloc[0],
                  "party": meta_df['member_party'].iloc[0],
                  "constituency": meta_df['member_constituency'].iloc[0],
                  "policy_positions": meta_df['policy_positions'].iloc[0],
                  "topic_assigned": meta_df['topic_assigned'].iloc[0]
                  }
            
        vectors_list.append(vector)

    return vectors_list

def prepare_bill_df_upsert(bill_meta_df, content):
    upsert_df = pd.merge(bill_meta_df,
            content.filter(['bill_number', 'bill_introduction', 'bill_key_points', 'bill_impact']), 
            'left', 
            'bill_number')

    # convert columns to accepted formats for pinecone
    upsert_df['date_introduced'] = upsert_df.date_introduced.astype(str)
    upsert_df['date_passed'] = upsert_df.date_passed.astype(str)
    upsert_df['parliament'] = upsert_df.parliament.astype(object)
    return upsert_df

def prepare_bill_data_upsert(bill_meta_df, bill_summaries, embed_dict):
    upsert_df = prepare_bill_df_upsert(bill_meta_df, bill_summaries)
    vectors_list = []
    for i in embed_dict:
        # get bill_number
        bill_number = i['custom_id']
        
        # get embeddings
        embeddings = list(i['response']['body']['data'][0]['embedding'])

        # get bill_summary and metadata
        meta_df = upsert_df.query(f"bill_number=='{bill_number}'")
        
        vector = {"id": bill_number, 
                "vector": embeddings,
                "title": meta_df['title'].iloc[0],
                "pdf_link": meta_df['pdf_link'].iloc[0],
                "date_introduced": meta_df['date_introduced'].iloc[0],
                "date_passed": meta_df['date_passed'].iloc[0],
                "parliament": meta_df['parliament'].iloc[0],
                "bill_introduction": meta_df['bill_introduction'].iloc[0],
                "bill_key_points": meta_df['bill_key_points'].iloc[0],
                "bill_impact": meta_df['bill_impact'].iloc[0]
                }
            
        vectors_list.append(vector)

    return vectors_list

def download_and_extract_bill_pdf(bill_link, pages = None):
    response = requests.get(bill_link, stream=True)
    with open(bill_pdf_directory, 'wb') as file:
          for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                     file.write(chunk)

    out_text = pymupdf4llm.to_markdown(bill_pdf_directory, pages = pages)
    return out_text

def count_bill_tokens(text):
    encoding = tiktoken.encoding_for_model(bills_model)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_bills_df_to_parts(df):
    df_list = []
    for ind,row in df.iterrows():
        full_text = row.bill_text
        increment = len(full_text)//row.bill_splits
        splits = 0
        split_df_list = []
        while splits!=row.bill_splits:
            start = int(splits*increment)
            stop = None if splits==row.bill_splits-1 else int((splits+1)*increment)
            split_text = full_text[start:stop]
            splits+=1
            split_df = df[['title', 'number']]
            split_df['split_text'] = split_text
            split_df['part'] = splits
            split_df_list.append(split_df)
            
        split_dfs = pd.concat(split_df_list)
        df_list.append(split_dfs)

    final_df = pd.concat(df_list)
    return final_df
    

