# run in bash
PINECONE_API_KEY=$(cat token/pinecone_token.txt)
OPENAI_API_KEY=$(cat token/gpt_api_token.txt)

@gcloud functions deploy generate_summaries --runtime python310 --trigger-http --allow-unauthenticated --entry-point generate_summaries --timeout=540s --memory=1GB --set-env-vars PINECONE_API_KEY="$PINECONE_API_KEY",OPENAI_API_KEY="$OPENAI_API_KEY"