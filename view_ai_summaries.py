import sqlite3
import pandas as pd

conn = sqlite3.connect("ai_support_knowledge.db")

# Read tables into DataFrames
tickets = pd.read_sql_query("SELECT * FROM tickets", conn)
summaries = pd.read_sql_query("SELECT * FROM ai_summaries", conn)

print("\n🧾 Tickets Table:")
print(tickets)

print("\n🤖 AI Summaries Table:")
print(summaries)

conn.close()
