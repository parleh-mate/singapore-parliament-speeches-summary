# run in bash
OPENAI_API_KEY=$(cat token/gpt_api_token.txt)
ZILLIZ_CLIENT_URI=$(cat token/zilliz_uri.txt)
ZILLIZ_API_KEY=$(cat token/zilliz_token.txt)

@gcloud functions deploy generate_summaries --runtime python310 --trigger-http --allow-unauthenticated --entry-point generate_summaries --timeout=540s --memory=1GB --set-env-vars ZILLIZ_API_KEY="$ZILLIZ_API_KEY",ZILLIZ_CLIENT_URI="$ZILLIZ_CLIENT_URI",OPENAI_API_KEY="$OPENAI_API_KEY"