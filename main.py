from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = openai_api_key)

#create an assistant that utilizes file_search tool (for reading pdfs)
assistant = client.beta.assistants.create(
  name="Text Relationships Extractor",
  instructions="You are an assistant that extracts specific relationships from research papers and provides output in a structured JSON format.",
  model="gpt-4o",
  tools=[{"type": "file_search"}],
)

# Create a vector store 
vector_store = client.beta.vector_stores.create(name="Clinical Research Papers")

article1 = "Articles/fava-et-al-2008-difference-in-treatment-outcome-in-outpatients-with-anxious-versus-nonanxious-depression-a-star_d-report (1).pdf"
article2 = "Articles/100-Papers-in-Clinical-Psychiatry-Depressive-Disorders-Comparative-efficacy-and-acceptability-of-12-new-generation-antidepressants-a-multiple-treatments-meta-analysis.pdf"
article3 = "Articles/WJCC-9-9350.pdf"
 
# Ready the files for upload to OpenAI
file_paths = [article1, article2, article3]
file_streams = [open(path, "rb") for path in file_paths]
 
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)


#connect the assistant with the vector store
assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# a Thread represents a conversation between a user and one or many Assistants.
thread = client.beta.threads.create()

# we can add a message or messages to the thread
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content= f"""
                Thoroughly extract the following relationships from the clinical research papers that have been provided:
                - Relationships betweens each symptom and disease
                - Relationships between each symptom and its treatment.
                - Relationships between each medication and the mental health problem it is effective against.
                - Relationships between each medication and its side-effect if available.
                

                Return the result in the following JSON format with at least 20 relationships:
                [
                    {{
                        "subject": "Entity1",
                        "relationship": "RelationshipType",
                        "object": "Entity2"
                    }}
                ]
                The output should just be in JSON. Do not add any additional words or messages
                """
)

# Once all the user Messages have been added to the Thread, you can Run the Thread with any Assistant.
run = client.beta.threads.runs.create_and_poll(
  thread_id=thread.id,
  assistant_id=assistant.id,
)

if run.status == 'completed': 
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    print(message_content.value)
else:
  print(run.status)

