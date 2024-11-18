from openai import OpenAI
from pyvis.network import Network
import networkx as nx
import json
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
                Imagine you are a tele mental health provider at an online mental health platform. You frequently see many
               patients with a variety of mental health disorders, particularly anxiety and depression. Your goal is to
               make quick, informed decisions about treatment plans and ensure patients receive the best care possible
               based on current clinical research.


               Using the provided clinical research papers, I need you to extract only the following key relationships to assist in making treatment decisions:
               - Relationships between each symptom and its associated mental health disorder (e.g., anxiety, depression).
               - Relationships between each symptom and its recommended treatment or intervention.
               - Relationships between each symptom and its recommended therapy.
               - Relationships between each medication and the mental health disorder it is most effective against.
               - Relationships between each medication and its known side effects, if available.


               Please return the results in the following JSON format, with at least 40 relationships:
               [
                   {{
                       "source": "Entity1",
                       "relationship": "RelationshipType",
                       "target": "Entity2",
                       "source_type": "3 possible classifications: Drug or Treatment, Condition or Symptom, Side Effect",
                       "target_type": "3 possible classifications: Drug or Treatment, Condition or Symptom, Side Effect"
                   }}
               ]
               The output should be provided in JSON format only. Do not include any additional words or messages. We are creating a knowledge after this with the JSON. 

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
    
    #normalize the relationships for some common variations 
    try:
        # parsing
        data = json.loads(message_content)

        relationship_mapping = {
            "recommended treatment for": "effective for",
            "recommended intervention for": "effective for",
            "adjunctive treatment for": "effective for",
            "effective for": "effective for",
            "side effects include": "side effect",
            "side effects": "side effect",
            "side effect": "side effect",
            "recommended therapy for": "treatment for",
            "combination therapy for": "treatment for",
        }

        # normalizing
        for relationship in data:
            relationship["relationship"] = relationship_mapping.get(
                relationship["relationship"], relationship["relationship"]
            )

        # saved normalized relationships
        normalized_relationships = data

        # verifying normalized relationships
        print(json.dumps(normalized_relationships, indent=4))

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON output: {e}")
    
else:
  print(run.status)



# message_content.value has extra characters
cleaned_content = message_content.value.strip('```json\n').strip('```')

#  parse the cleaned JSON content
try:
    edges = json.loads(cleaned_content)
    print("JSON Parsed Successfully!")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")

print(edges)


# networkX directed graph
G = nx.DiGraph()

# three possible types - drug / treatment, condition / symptoms, side effect
# feel free to play around with the colors
color_map = {
    "Drug / Treatment": "green",  
    "Condition / Symptom": "lightcoral",    
    "Side Effect": "lightblue"
}

shape_map = {
    "Drug / Treatment": "dot",      
    "Condition / Symptom": "dot",     
    "Side Effect": "dot"        
}


# add nodes and edges to the graph 
for edge in edges:
    
    G.add_node(edge['source'], title=edge['source'], color=color_map.get(edge['source_type'], 'gray'), shape= shape_map.get(edge['source_type'], 'star')) 
    G.add_node(edge['target'], title=edge['target'], color=color_map.get(edge['target_type'], 'gray'), shape=shape_map.get(edge['target_type'], 'star'))
    G.add_edge(edge['source'], edge['target'], label=edge['relationship'], color="#b4b7b8")

# create the PyVis Network object
net = Network(height="1200px", width="100%", notebook=True, directed=True, cdn_resources='in_line')

# load the NetworkX graph into PyVis
net.from_nx(G)
net.set_options("""
{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -30,
      "centralGravity": 0.005,
      "springLength": 200,
      "springConstant": 0.01
    },
    "minVelocity": 0.75,
    "stabilization": {
      "enabled": true,
      "iterations": 150
    }
  }
}
""")


"""
DOCUMENTATION: SET_OPTIONS
net.set_options(

   "physics": {
     - Enables a physics engine to control how nodes move and interact.
     - This section adjusts repulsion, attraction, and stabilization, 
       affecting how the graph is laid out dynamically.

     "solver": "forceAtlas2Based"
       - Specifies the physics model used to arrange nodes.
       - "forceAtlas2Based" balances the forces between nodes:
         * Repulsive forces push nodes apart to reduce clutter.
         * Attractive forces draw connected nodes closer together.

     "forceAtlas2Based": {
       - Configures parameters for the "forceAtlas2Based" model:

       "gravitationalConstant": -30
         * A negative value increases node repulsion.
         * Higher absolute values lead to a more spread-out graph.

       "centralGravity": 0.005
         * Controls the tendency of nodes to move toward the graph's center.
         * Lower values lead to a more dispersed node arrangement.

       "springLength": 200
         * Defines the ideal distance between connected nodes.
         * Larger values make connected nodes farther apart.

       "springConstant": 0.01
         * Sets the stiffness of the connection between nodes.
         * Lower values provide more flexibility, allowing nodes to 
           move freely before reaching a stable state.
     }

     "minVelocity": 0.75
       - Sets the minimum movement speed for nodes.
       - Higher values lead to faster stabilization of the graph.

     "stabilization": {
       - Adjusts how the graph settles into a stable layout.

       "enabled": true
         * Activates stabilization, allowing nodes to adjust 
           positions until the layout becomes stable.

       "iterations": 150
         * Specifies the number of stabilization iterations.
         * More iterations allow for finer adjustments, leading to 
           a more stable and visually organized graph.
     }
   }
---------------------------------------------------------------
"""

# generate and display the interactive graph in a notebook
net.show("medication_condition_relationships.html")

