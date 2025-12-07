from openai import OpenAI
from google.cloud import bigquery
from params.api_keys import *
import os
import google.auth
from pymilvus import MilvusClient

from summarize_positions import handle_positions_extraction, handle_positions_creation
from summarize_bills import handle_bill_summaries_extraction, handle_bill_summaries_creation, check_bill_updates

# initialize clients
gpt_client = OpenAI()


SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive.readonly",
]
creds, project_id = google.auth.default(scopes=SCOPES)
gbq_client = bigquery.Client(project=project_id, credentials=creds)

zilliz_client = MilvusClient(
    uri=os.getenv("ZILLIZ_CLIENT_URI"),
    token=os.getenv("ZILLIZ_API_KEY"), 
)

def generate_summaries(request):
    """Cloud Function entry point to generate summaries."""
    try:
        # start with speech positions
        handle_positions_extraction(gbq_client, gpt_client, zilliz_client)
        handle_positions_creation(gbq_client, gpt_client)

        # handle summaries for bills
        handle_bill_summaries_extraction(gbq_client, gpt_client, zilliz_client)
        handle_bill_summaries_creation(gbq_client, gpt_client)

        # check bill updates
        check_bill_updates(gbq_client, zilliz_client)
        return 'Summaries run successfully.', 200
    except Exception as e:
        print(f"Error running function: {e}")
        return f"Error: {e}", 500