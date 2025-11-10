import os
from groq import Groq

# Initialize client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Example support ticket
ticket = """
Hello team, I'm unable to log into my Effivity account since yesterday.
It keeps saying 'Invalid user credentials' even though my password is correct.
I tried resetting it, but I never got the reset email.
Can you please help urgently? We have an audit scheduled tomorrow.
"""

prompt = f"""
You are an expert technical support assistant.
Read the following customer ticket and summarize it clearly:
1. Problem Summary
2. Possible Root Cause
3. Recommended Action

Ticket:
{ticket}
"""

# Use an open model (like LLaMA 3)
response = client.chat.completions.create(
model="llama-3.1-8b-instant",

    messages=[{"role": "user", "content": prompt}]
)

print("📝 AI Summary:\n")
print(response.choices[0].message.content)
