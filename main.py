import pandas as pd
from openai import OpenAI
from datetime import date
from google.cloud import bigquery

from extract import *
from load import *
from utils import *
from params import *

# gpt client
gpt_client = OpenAI()

# GBQ client
gbq_client = bigquery.Client()

# find in progress jobs
batch_check_df = get_in_progress_jobs(gbq_client, dim_table_id)

# get batch ids
gpt_batch_id = batch_check_df.gpt_batch_id[0]
batch_id = batch_check_df.batch_id[0]

# only continue if a job was scheduled
if (len(batch_check_df)!=0):

    batch_meta = retrieve_batch_meta(gpt_batch_id, gpt_client)

    # only extract batch data if the job is completed
    if batch_meta.status=="completed":
        speech_summaries = extract_summarized_data(batch_meta, gpt_client, batch_id, gpt_batch_id, topics_list)

        # now upload to gbq
        create_or_append_table(speech_summaries, summaries_table_id, gbq_client)

# now start creating new jobs

unsummarized_df = get_unsummarized_speeches(gbq_client, upper_bound, lower_bound, batch_size)

# now create batch file from unsummarized speeches (writes locally)
create_json_batch_file(unsummarized_df, model, system_message, response_format, local_batch_directory)

# now upload this batch to gpt and get the batch_id
gpt_batch_id = upload_batch_to_gpt(gpt_client, local_batch_directory)

# new batch id and date
batch_id = batch_id + 1
batch_date = date.today()

# ready new batch meta to be uploaded to gbq
dim_speech_summaries = pd.DataFrame({"batch_id": [batch_id],
                                     "gpt_batch_id": [gpt_batch_id],
                                     "model": [model],
                                     "batch_date": [batch_date],
                                     "system_message": [system_message],
                                     "output_summary_description": [output_summary_description],
                                     "output_topic_description": [output_topic_description],
                                     "lower_word_bound": [lower_bound],
                                     "upper_word_bound": [upper_bound],
                                     "word_limit": [word_limit],
                                     "status": ["in_progress"]})

# upload batch meta to gbq
create_or_append_table(dim_speech_summaries, dim_table_id, gbq_client, gbq_dim_schema)




    