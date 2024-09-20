from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = openai_api_key)

completion = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are the a historian."},
        {"role": "user", "content": "What was the most important battle of World War 2? Keep your responses short and straightforward yet insightful and compelling"}
    ]
)

print(completion.choices[0].message.content)
