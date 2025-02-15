from openai import OpenAI
from google.cloud import bigquery
from pinecone import Pinecone
#from params.api_keys import *

from summarize_positions import handle_positions_extraction, handle_positions_creation
from summarize_bills import handle_bill_summaries_extraction, handle_bill_summaries_creation, check_bill_updates

# initialize clients
gpt_client = OpenAI()
gbq_client = bigquery.Client()
pc = Pinecone()
positions_index = pc.Index("singapore-speeches-positions")
summaries_index = pc.Index("singapore-bill-summaries")

def generate_summaries(request):
    """Cloud Function entry point to generate summaries."""
    try:
        # start with speech positions
        handle_positions_extraction(gbq_client, gpt_client, positions_index)
        handle_positions_creation(gbq_client, gpt_client)

        # handle summaries for bills
        handle_bill_summaries_extraction(gbq_client, gpt_client, summaries_index)
        handle_bill_summaries_creation(gbq_client, gpt_client)

        # check bill updates
        check_bill_updates(gbq_client, summaries_index)
        return 'Summaries run successfully.', 200
    except Exception as e:
        print(f"Error running function: {e}")
        return f"Error: {e}", 500