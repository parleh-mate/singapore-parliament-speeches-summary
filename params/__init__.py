import os
import pandas as pd
from google.cloud import bigquery

# word limit bounds
upper_bound = 2000
lower_bound = 70
batch_size = 1000
word_limit = 150

# get topics

topics_list = list(pd.read_csv("assets/topics_LDA.csv").query("include").topic_name.unique())
topics = ", ".join(topics_list)

# gpt prompts

system_message = f"""You will be provided with a speech from the Singapore parliament. You are a helpful assistant who will summarize this speech in no more than ~{word_limit} words and label it with a topic from a set list. Do not hallucinate new topics; you are to label a speech ONLY from one of these topics: [{topics}]
"""

output_summary_description = f"""A concise summary of the speech of no more than {word_limit} words. Sometimes speeches are long and unsubstantive. I will provide a step-by-step process to guide your summarization:

1. Extract all the key policy points in the speech and omit anything that is unsubstantive. Unsubstantive items include reiteration of someone else's speech, parliamentary decorum (thanking another member or the speaker), and procedural points. 

2. Rank the key points according to this hierarchy, where 1 indicates highest priority: 1) specific policy proposal or recommendation 2) advocacy or highlighting an issue without mentioning specific remedies 3) miscellaneous commentary or anecdotes

3. Summarize the speech and give priority to items higher up in the hierarchy, and if need be generously omit things that cannot fit into the word limit.

Note that {word_limit} words is only a limit and a summary can be a lot shorter if the speech is short and unsubstantive. As a rule of thumb a summary cannot be longer than the speech it is summarizing.

Adhere strictly to the following writing style: 

1. Use concise language, avoiding tautology. 

2. Write in the present tense passive voice like you would an objective report, avoiding pronouns if possible. For example, 'Singapore has longstanding partnerships with China' is preferred to 'I highlight Singapore's longstanding partnerships with China'.

3. If you have to use pronouns, strictly avoid using the first-person and write in the third-person instead. For example, 'The speaker/speech emphasizes the need to support SMEs during pandemic recovery' is preferred to 'We need to support SMEs during pandemic recovery'.

4. Do not expand acronyms, just leave them as they are.
"""

output_topic_description = f"""The topic of the speech chosen ONLY from one from these topics: [{topics}]. 
    
Some speeches are responses to a parliamentary question. In this case, I will provide the ministry(s) to which the question is addressed to help you with the labelling. In the case that there are two, the first ministry will be the ministry that is mentioned the most, and the second one addressed second-most. Use this information wisely since speeches may digress from the initial question.
"""

# gpt structured formats output

response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": {"type": "object", "properties": {"Summary": {"type": "string", "description": output_summary_description}, "Topic": {"type": "string", "description": output_topic_description}}, "required": ["Summary", "Topic"], "additionalProperties": False}}}

# gpt model type
model = "gpt-4o-mini"

# where should the batch file be saved locally?

local_batch_directory = 'assets/batch_summary.jsonl'

# gbq summaries schema

gbq_dim_schema = [
    bigquery.SchemaField("batch_id", "INTEGER"),
    bigquery.SchemaField("gpt_batch_id", "STRING"),
    bigquery.SchemaField("model", "STRING"),
    bigquery.SchemaField("batch_date", "DATE"),
    bigquery.SchemaField("system_message", "STRING"),
    bigquery.SchemaField("output_summary_description", "STRING"),
    bigquery.SchemaField("output_topic_description", "STRING"),
    bigquery.SchemaField("lower_word_bound", "INTEGER"),
    bigquery.SchemaField("upper_word_bound", "INTEGER"),
    bigquery.SchemaField("word_limit", "INTEGER"),
    bigquery.SchemaField("status", "STRING")
]

# gcp key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "token/gcp_token.json"

# gpt key
os.environ["OPENAI_API_KEY"] = open("token/gpt_api_token.txt", 'r').readlines()[0]

# pinecone api key
os.environ['PINECONE_API_KEY'] = open("token/pinecone_token.txt", 'r').readlines()[0]

# table names
dim_table_id = "singapore-parliament-speeches.prod_dim.dim_speech_summaries"
dim_embeddings_table_id = "singapore-parliament-speeches.prod_dim.dim_speech_embeddings"
summaries_table_id = "singapore-parliament-speeches.prod_mart.mart_speech_summaries"
speeches_table_id = "singapore-parliament-speeches.prod_mart.mart_speeches"