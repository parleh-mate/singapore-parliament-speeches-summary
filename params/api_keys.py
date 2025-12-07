import os

# gcp key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "token/gcp_token.json"

# gpt key
os.environ["OPENAI_API_KEY"] = open("token/gpt_api_token.txt", 'r').readlines()[0]

# zilliz api key
os.environ['ZILLIZ_API_KEY'] = open("token/zilliz_token.txt", 'r').readlines()[0]

# zilliz client uri
os.environ['ZILLIZ_CLIENT_URI'] = open("token/zilliz_uri.txt", 'r').readlines()[0]