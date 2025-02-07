import pandas as pd
from google.cloud import bigquery

# job descriptions
speech_embeddings_job_description = "singapore parliamentary speech embeddings batch job"
bill_embeddings_job_description = "singapore parliamentary bill embeddings batch job"

# word limit bounds
word_upper_bound = 2000
word_lower_bound = 70

# batch sizes
speech_batch_size = 200
bills_batch_size = 5

# get topics
topics_list = list(pd.read_csv("assets/topics_LDA.csv").query("include").topic_name.unique())
topics = ", ".join(topics_list)

# gpt model type
positions_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
bills_model = "gpt-4o-mini"
embedding_dimensions = 1536

# where should the batch file be saved locally?

local_positions_batch_directory = 'assets/batch_positions_summary.jsonl'
local_bills_batch_directory = 'assets/batch_bills_summary.jsonl'
local_bills_split_batch_directory = 'assets/batch_bills_split_summary.jsonl'
bill_pdf_directory = 'assets/bill.pdf'

# schemas

def gbq_positions_embeddings_schema(content_id_var_name):
    return [
        bigquery.SchemaField("batch_id", "INTEGER"),
        bigquery.SchemaField("gpt_batch_id", "STRING"),
        bigquery.SchemaField("model", "STRING"),
        bigquery.SchemaField("dimensions", "INTEGER"),
        bigquery.SchemaField("batch_date", "DATE"),
        bigquery.SchemaField(content_id_var_name, "STRING", mode="REPEATED"),
        bigquery.SchemaField("status", "STRING")
    ]

gbq_positions_schema = [
    bigquery.SchemaField("batch_id", "INTEGER"),
    bigquery.SchemaField("gpt_batch_id", "STRING"),
    bigquery.SchemaField("model", "STRING"),
    bigquery.SchemaField("batch_date", "DATE"),
    bigquery.SchemaField("system_message", "STRING"),
    bigquery.SchemaField("output_position_description", "STRING"),
    bigquery.SchemaField("output_topic_description", "STRING"),
    bigquery.SchemaField("lower_word_bound", "INTEGER"),
    bigquery.SchemaField("upper_word_bound", "INTEGER"),
    bigquery.SchemaField("status", "STRING")
]

gbq_bill_summaries_schema = [
    bigquery.SchemaField("batch_id", "INTEGER"),
    bigquery.SchemaField("gpt_batch_id", "STRING"),
    bigquery.SchemaField("model", "STRING"),
    bigquery.SchemaField("batch_date", "DATE"),
    bigquery.SchemaField("system_message", "STRING"),
    bigquery.SchemaField("output_bill_introduction_description", "STRING"),
    bigquery.SchemaField("output_bill_key_points_description", "STRING"),
    bigquery.SchemaField("output_bill_impact_description", "STRING"),
    bigquery.SchemaField("status", "STRING")
]

gbq_bill_split_summaries_schema = [
    bigquery.SchemaField("batch_id", "INTEGER"),
    bigquery.SchemaField("gpt_batch_id", "STRING"),
    bigquery.SchemaField("model", "STRING"),
    bigquery.SchemaField("batch_date", "DATE"),
    bigquery.SchemaField("system_message", "STRING"),
    bigquery.SchemaField("bills_split_key_points_description", "STRING"),
    bigquery.SchemaField("bill_numbers", "STRING", mode="REPEATED"),
    bigquery.SchemaField("bill_splits", "INTEGER", mode="REPEATED"),
    bigquery.SchemaField("status", "STRING")
]