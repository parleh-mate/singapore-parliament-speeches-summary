import os

# gcp key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "token/gcp_token.json"

# gpt key
os.environ["OPENAI_API_KEY"] = open("token/gpt_api_token.txt", 'r').readlines()[0]

# pinecone api key
os.environ['PINECONE_API_KEY'] = open("token/pinecone_token.txt", 'r').readlines()[0]