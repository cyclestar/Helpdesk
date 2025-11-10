import pandas as pd
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Read tickets from Excel file
df = pd.read_excel("sample_tickets.xlsx")

for _, t in df.iterrows():
    prompt = f"""
    You are a senior tech support assistant.
    Summarize this ticket briefly:
    1. Problem Summary
    2. Possible Root Cause
    3. Recommended Action

    Ticket:
    {t['Issue']}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response.choices[0].message.content.strip()
    print(f"\n🧾 Ticket {t['TicketID']} - {t['Customer']}")
    print(summary)
    print("-" * 60)
