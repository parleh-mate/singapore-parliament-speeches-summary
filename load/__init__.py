from google.cloud import bigquery
import pandas as pd
from datetime import date

from extract import get_last_job
from params import embedding_model, embedding_dimensions, gbq_positions_embeddings_schema


def create_or_append_table(df, table_id, gbq_client, schema = None):
    try:
        gbq_client.get_table(table_id)  # Check if table exists
        print(f"Table {table_id} exists, appending data.")
        # If the table exists, configure the load job to append data
        job_config = bigquery.LoadJobConfig(
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND  # Append to existing table
        )
    except Exception as e:
        print(f"Table {table_id} does not exist, creating table.")
        # If the table does not exist, create it and upload data
        table = bigquery.Table(table_id, schema = schema)
        gbq_client.create_table(table)  # Create the table
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_EMPTY  # Create table if empty
        )

    job = gbq_client.load_table_from_dataframe(df,
                                               table_id,
                                               job_config = job_config)
    job.result()

def upload_embedding_meta_gbq(gbq_client, table_id, gpt_batch_id, content_id_var_name, content_ids):
    embeddings_batch_check_df = get_last_job(gbq_client, table_id)
    embedding_batch_id = embeddings_batch_check_df.batch_id[0]

    # new batch id and date
    embedding_batch_id = embedding_batch_id + 1
    batch_date = date.today()

    # ready new batch meta to be uploaded to gbq
    dim_embeddings = pd.DataFrame({"batch_id": [embedding_batch_id],
                                        "gpt_batch_id": [gpt_batch_id],
                                        "model": [embedding_model],
                                        "dimensions": [embedding_dimensions],
                                        "batch_date": [batch_date],
                                        content_id_var_name: [content_ids.tolist()],
                                        "status": ["completed"]})  

    gbq_schema = gbq_positions_embeddings_schema(content_id_var_name)      

    # upload embeddings batch meta to gbq
    create_or_append_table(dim_embeddings, table_id, gbq_client, gbq_schema)