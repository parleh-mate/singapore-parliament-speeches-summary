from google.cloud import bigquery

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