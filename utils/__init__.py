import jsonlines

def change_to_completed(gbq_client, dim_table_id):
    job = gbq_client.query(f"""
                            UPDATE `{dim_table_id}`
                            SET status = 'completed'
                            WHERE status = 'in_progress'
                            """)
    job.result()

def create_json_batch_file(df, model, system_message, response_format, local_batch_directory):
    json_list = []
    for ind,row in df.iterrows():

        ministries = [i for i in row.filter(['ministry_addressed_primary', 
                                                            'ministry_addressed_secondary']) if i is not None]
        
        mins_addressed = ','.join(ministries)

        input = f"Speech: [{row.speech_text}], [Ministries addressed: {mins_addressed}]"    

        json_list.append({"custom_id": row.speech_id, 
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, 
                "messages": [{"role": "system", 
                                "content": system_message},
                                {"role": "user", 
                                "content": input}],
                                "max_tokens": 3000,
                                "response_format": response_format
        }})

    with jsonlines.open(local_batch_directory, 'w') as writer:
        writer.write_all(json_list)


def upload_batch_to_gpt(gpt_client, local_batch_directory):
    batch_input_file = gpt_client.files.create(
    file=open(local_batch_directory, "rb"),
    purpose="batch"
    )

    batch_file_id = batch_input_file.id

    batch_meta = gpt_client.batches.create(
        input_file_id=batch_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "singapore parliamentary speech summary batch job"
        }
    )

    return batch_meta.id
