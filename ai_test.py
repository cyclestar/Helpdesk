import os
from openai import OpenAI

# Initialize client using the environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Send a simple test prompt to GPT
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain AI in one simple sentence."}
    ]
)

# Print the model's reply
print("🤖 GPT says:", response.choices[0].message.content)
