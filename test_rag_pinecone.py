from google.cloud import bigquery
from pinecone import Pinecone
from openai import OpenAI
import jsonlines
import json
from datetime import date
import time

from params import *
from load import create_or_append_table
from extract import get_in_progress_jobs, retrieve_batch_meta

# GBQ client
gbq_client = bigquery.Client()

# gpt client
gpt_client = OpenAI()

# pinecone client
pc = Pinecone()
index = pc.Index("singapore-summarized-speeches")

pause_time = 120

while True:
    print("starting loop")

    # check if previous job was completed
    batch_check_df = get_in_progress_jobs(gbq_client, dim_embeddings_table_id)
    # get batch ids
    gpt_batch_id = batch_check_df.gpt_batch_id[0]
    batch_id = batch_check_df.batch_id[0]

    # only continue if a job was scheduled
    if (batch_check_df.status.iloc[0]=='in_progress'):

        batch_meta = retrieve_batch_meta(gpt_batch_id, gpt_client)

        print("checking status of previous batch")

        # only extract batch data if the job is completed
        if batch_meta.status=="completed":
            print("previous batch completed")

            # extract embedded data
            file_response = gpt_client.files.content(batch_meta.output_file_id)
            embed_response = file_response.read().decode('utf-8')
            embed_dict = [json.loads(obj) for obj in embed_response.splitlines()]

            # collect speech summaries from gbq with speech metadata to be used for upserting later
            job = gbq_client.query(f"""
                            SELECT speech_id, `date`, parliament, member_name, member_party, member_constituency, speech_summary, topic_assigned
                            FROM `{summaries_table_id}`
                            left join (select speech_id, member_name, member_party, member_constituency, parliament, `date` from `{speeches_table_id}`)
                            using (speech_id)
                            WHERE speech_id IN (
                                SELECT B_speech_id
                                FROM `{dim_embeddings_table_id}` AS B,
                                UNNEST(B.speech_ids) AS B_speech_id
                                WHERE status = "in_progress"
                            )
                            """)

            result = job.result()
            summaries_df = result.to_dataframe()

            # convert columns to accepted formats for pinecone
            summaries_df['date'] = summaries_df.date.astype(str)
            summaries_df['parliament'] = summaries_df.parliament.astype(object)

            # now prepare data in correct format for upserting to pinecone

            vectors_list = []
            for i in embed_dict:
                # get speech_id
                speech_id = i['custom_id']
                
                # get embeddings
                embeddings = list(i['response']['body']['data'][0]['embedding'])

                # get speech_summary and metadata
                summary_df = summaries_df.query(f"speech_id=='{speech_id}'")
                
                vector = {"id": speech_id, 
                        "values": embeddings,
                        "metadata": {"date": summary_df['date'].iloc[0],
                                     "parliament": summary_df['parliament'].iloc[0],
                                     "name": summary_df['member_name'].iloc[0],
                                     "party": summary_df['member_party'].iloc[0],
                                     "constituency": summary_df['member_constituency'].iloc[0],
                                     "summary": summary_df['speech_summary'].iloc[0],
                                     "topic_assigned": summary_df['topic_assigned'].iloc[0]
                                     }}
                
                vectors_list.append(vector)

            # Upsert to pinecone
            index.upsert(
                vectors = vectors_list
            )
            print("vectors upserted")

            # now change executing entries
            job = gbq_client.query(f"""
                            UPDATE `{dim_embeddings_table_id}`
                            SET status = 'completed'
                            WHERE status = 'in_progress'
                            """)
            job.result()

        elif (batch_meta.status=='failed'):

            print("Job failed, deleting previous job")
            # if job failed, delete in progress entry and go immediately to creating new batch
            job = gbq_client.query(f"""
                            DELETE 
                            FROM `{dim_embeddings_table_id}`
                            WHERE status = 'in_progress'
                            """)
            job.result()

        else:
            print("Previous batch job not completed, restarting loop")
            print(f"starting {pause_time}s pause")
            time.sleep(pause_time)
            print("pause ended")
            continue       

    # now start creating new jobs
    # get speech ids of speeches not yet embedded

    print("Creating new batch")

    job = gbq_client.query(f"""
                            SELECT speech_id, speech_summary
                            FROM `{summaries_table_id}`
                            WHERE speech_id NOT IN (
                                SELECT B_speech_id
                                FROM `singapore-parliament-speeches.prod_dim.dim_speech_embeddings` AS B,
                                UNNEST(B.speech_ids) AS B_speech_id
                            )
                            limit 200
                            """)

    result = job.result()
    summaries_df = result.to_dataframe()
    speeches = summaries_df.speech_summary
    speech_ids = summaries_df.speech_id

    # create batch job for text embeddings

    model = "text-embedding-3-small"
    embedding_dimensions = 1536

    json_list = []
    for i in range(len(speeches)): 

        json_list.append({"custom_id": speech_ids[i], 
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"input": speeches[i],
                "model": model,
                "encoding_format": "float",
                'dimensions': embedding_dimensions
                }})

    with jsonlines.open("assets/batch_embeddings.jsonl", 'w') as writer:
        writer.write_all(json_list)

    # embeddings = gpt_client.embeddings.create(
    #     input = speeches,
    #     model = "text-embedding-3-small"
    #     )

    # upload batch job

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
        "description": "singapore parliamentary speech embeddings batch job"
        }
    )

    gpt_batch_id = batch_meta.id

    # upload job id to gbq
    # new batch id and date
    batch_id = batch_id + 1
    batch_date = date.today()

    # ready new batch meta to be uploaded to gbq
    dim_speech_embeddings = pd.DataFrame({"batch_id": [batch_id],
                                        "gpt_batch_id": [gpt_batch_id],
                                        "model": [model],
                                        "dimensions": [embedding_dimensions],
                                        "batch_date": [batch_date],
                                        "speech_ids": [speech_ids],
                                        "status": ["in_progress"]})

    gbq_dim_schema = [
        bigquery.SchemaField("batch_id", "INTEGER"),
        bigquery.SchemaField("gpt_batch_id", "STRING"),
        bigquery.SchemaField("model", "STRING"),
        bigquery.SchemaField("dimensions", "INTEGER"),
        bigquery.SchemaField("batch_date", "DATE"),
        bigquery.SchemaField("speech_ids", "STRING", mode="REPEATED"),
        bigquery.SchemaField("status", "STRING")
    ]

    # upload batch meta to gbq
    create_or_append_table(dim_speech_embeddings, dim_embeddings_table_id, gbq_client, gbq_dim_schema)

    print("Batch uploaded, end of loop")
    print(f"Start {pause_time}s pause")
    time.sleep(pause_time)
    print("pause ended, restarting loop")