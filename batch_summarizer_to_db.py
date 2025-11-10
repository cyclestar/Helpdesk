import pandas as pd
import os
import sqlite3
from groq import Groq

# -----------------------------
# 1. Setup
# -----------------------------
DB_PATH = "ai_support_knowledge.db"
EXCEL_PATH = "sample_tickets.xlsx"

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 2. Read Excel data
# -----------------------------
df = pd.read_excel(EXCEL_PATH)

# -----------------------------
# 3. Process each ticket
# -----------------------------
for _, row in df.iterrows():
    customer = row["Customer"]
    issue = row["Issue"]

    # Insert original ticket into database
    cursor.execute(
        "INSERT INTO tickets (customer_name, issue_text) VALUES (?, ?)",
        (customer, issue)
    )
    ticket_id = cursor.lastrowid

    # Create AI prompt
    prompt = f"""
    You are a senior tech support assistant.
    Summarize this ticket clearly:
    1. Problem Summary
    2. Possible Root Cause
    3. Recommended Action

    Ticket:
    {issue}
    """

    # Get AI response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    ai_output = response.choices[0].message.content.strip()

    # -----------------------------
    # 4. Save AI result to database
    # -----------------------------
    cursor.execute(
        "INSERT INTO ai_summaries (ticket_id, ai_summary, model_used) VALUES (?, ?, ?)",
        (ticket_id, ai_output, "llama-3.1-8b-instant")
    )

    print(f"✅ Ticket {ticket_id} processed for {customer}")

# Commit and close
conn.commit()
conn.close()

print("\n🎉 All tickets processed and stored in ai_support_knowledge.db")
