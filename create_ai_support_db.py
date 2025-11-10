import sqlite3

# Connect (creates file if not exists)
conn = sqlite3.connect("ai_support_knowledge.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT,
    issue_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS ai_summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER,
    ai_summary TEXT,
    root_cause TEXT,
    recommendation TEXT,
    model_used TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS resolutions (
    resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER,
    resolution_text TEXT,
    resolved_by TEXT,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER,
    rating INTEGER,
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
);
""")

conn.commit()
conn.close()

print("✅ Database and tables created successfully!")
