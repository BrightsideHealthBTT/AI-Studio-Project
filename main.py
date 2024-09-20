from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = openai_api_key)

assistant = client.beta.assistants.create(
  name="Text Relationships Extractor",
  instructions="You are an assistant that extracts specific relationships from research papers and provides output in a structured JSON format.",
  model="gpt-4o",
  tools=[{"type": "file_search"}],
)

# Create a vector store 
vector_store = client.beta.vector_stores.create(name="Clinical Research Papers")
 
# Ready the files for upload to OpenAI
file_paths = ["Articles/fava-et-al-2008-difference-in-treatment-outcome-in-outpatients-with-anxious-versus-nonanxious-depression-a-star_d-report (1).pdf"]
file_streams = [open(path, "rb") for path in file_paths]
 
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)


# Upload the user provided file to OpenAI
message_file = client.files.create(
  file=open("Articles/fava-et-al-2008-difference-in-treatment-outcome-in-outpatients-with-anxious-versus-nonanxious-depression-a-star_d-report (1).pdf", "rb"), purpose="assistants"
)
 
# Create a thread and attach the file to the message
thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": f"""
                Thoroughly extract the following relationships from the text:
                - Relationships betweens symptoms and diseases
                - Relationships between medications and outcomes.
                - Relationships between symptoms and treatments.
                - Relationships between medications and side effects.
                I want as many useful relationships as possible. 
                Return the result in the following JSON format:

                [
                    {{
                        "subject": "Entity1",
                        "relationship": "RelationshipType",
                        "object": "Entity2"
                    }}
                ]""",
      # Attach the new file to the message.
      "attachments": [
        { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
      ],
    }
  ]
)
 
# The thread now has a vector store with that file in its tool resources.
print(thread.tool_resources.file_search)


# Use the create and poll SDK helper to create a run and poll the status of
# the run until it's in a terminal state.

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

message_content = messages[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))