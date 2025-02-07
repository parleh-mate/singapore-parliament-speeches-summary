from params import topics

# gpt prompts

# positions summary

positions_summary_system_message = f"""You will be provided with a speech from the Singapore parliament. You are a helpful assistant who will extract the speaker's policy positions from this speech. This is a crucial step in a multi-stage process, where the final product is an API that can summarize a politician or party's policy position on a user-submitted query. Policy positions will be transformed into word embeddings, which will later be queried using a RAG model and an LLM will summarize these policies into a coherent policy position.

At the same time, you will also label the speech with a topic from a set list. Do not hallucinate new topics; you are to label a speech ONLY from one of these topics: [{topics}]
"""

output_position_description = f"""A list of policy positions expressed by the speaker from this speech. Here is a step to step guide on how you should do this.

1. Identify policy positions that the speaker expressed in the speech. A policy position can be when the speaker expresses support for an existing policy, suggests a new policy, or simply voices their opinion on a given issue. 

2. Omit anything that is unsubstantive. Unsubstantive items include reiteration of someone else's speech, parliamentary decorum (thanking another member or the speaker), and procedural points. If a speech has no policy positions, simply return 'no positions found'.

3. Cross reference each position to ensure that they are those advocated for by the speaker. In many cases, a speaker may mention an existing policy or one suggested by another speaker, but not necessarily agree with it. They may mention it to segue into a different topic, to ask questions about it, or even to disagree with it. This should not be included in the speaker's list of positions, since they did not advocate for it. Remember that your job is to list positions that the speaker expressed, not simply to summarize what they said.

Adhere strictly to the following writing style: 

1. Use concise language but liberally include specifics of policies if they are present, rather than summarizing them generically. If necessary, always choose detail over succinctness.

2. Do not expand acronyms, just leave them as they are.

3. Output should be a bullet list with '-' used as the bullet.

4. State the policy position directly rather than use action words. For example, 'Emphasize the importance of supporting youths financially' should simply be 'support youths financially'. Remember that the output is not a summary of the speech but a list of policy positions.
"""

output_topic_description = f"""The topic of the speech chosen ONLY from one from these topics: [{topics}]. 
"""

positions_summary_response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": {"type": "object", "properties": {"Positions": {"type": "string", "description": output_position_description}, "Topic": {"type": "string", "description": output_topic_description}}, "required": ["Positions", "Topic"], "additionalProperties": False}}}

# bill summaries

bills_summary_system_message = """You will be provided with a parliamentary bill from the Singapore parliament. You are a helpful assistant who will write a short description of the bill, summarize the bill into key bullet points, and explain its direct impact on citizens.
"""

output_bill_introduction_description = f"""A short description of what the bill is about in no more than 2-3 sentences. Output should be concise and free of technical jargon but not overly general. Include detailed specifics if need be.
"""

output_bill_key_points_description = f"""A summary of 5-10 key points from the parliamentary bill, with similar policy changes grouped together into a single point.

Adhere strictly to the following writing style:

1. Avoid legal jargon and explain key points in ways that a lay person would understand. 

2. Include specifics rather than over generalizing. For example 'income tax for citizens with a household income below $3000 will be lowered from 20-18%' is preferred to 'income tax for low income households will be lowered' .If necessary, always choose detail over succinctness.

3. Begin bullet points with action words.

4. Do not expand acronyms, just leave them as they are.

5. Output should be a bullet list with '-' used as the bullet.
"""

output_bill_impact_description = f"""A short summary of the bill's impact on citizens and businesses. Remain neutral and objective, considering only what the bill will directly do as opposed to what it intends to do and other potential repurcussions. 
"""

bills_summary_response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": {"type": "object", "properties": {"bill_introduction": {"type": "string", "description": output_bill_introduction_description}, "bill_key_points": {"type": "string", "description": output_bill_key_points_description}, "bill_impact": {"type": "string", "description": output_bill_impact_description}}, "required": ["bill_introduction", "bill_key_points", "bill_impact"], "additionalProperties": False}}}

# split bills into smaller portions
bills_split_summary_system_message = f"""You will be provided with a part of a parliamentary bill from the Singapore parliament. The bill has been split into parts because it is too long. You are a helpful assistant who summarize this part of the bill into key points which will later be combined with the summaries of other parts to be resummarized again.
"""

bills_split_key_points_description = f"""Key points from the parliamentary bill.

Adhere strictly to the following writing style:

1. Because the output will be summarized again, retain as much detailed information as possible while keeping sentences concise.

2. Group similar points together into a single point.

3. You are allowed to use as many tokens as allowed by the model's output limit.

4. Note that not all details need to be included, only those that are most important. Importance is determined by which has the most significant and direct impact on citizens. Use this to decide which points to keep in order to keep within the maximum output token limit.

5. Output should be a bullet list with '-' used as the bullet.
"""

bills_split_summary_response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": {"type": "object", "properties": {"bill_split_key_points": {"type": "string", "description": bills_split_key_points_description}}, "required": ["bill_split_key_points"], "additionalProperties": False}}}



