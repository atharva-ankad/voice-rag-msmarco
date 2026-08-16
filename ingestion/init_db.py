import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag_sidecar.db')

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS passages (
        passage_id TEXT PRIMARY KEY,
        query_id TEXT NOT NULL,
        source_language TEXT NOT NULL,
        target_language TEXT NOT NULL,
        english_text TEXT,
        translated_text TEXT,
        is_selected INTEGER NOT NULL,
        split TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        passage_id TEXT NOT NULL,
        strategy TEXT NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        FOREIGN KEY(passage_id) REFERENCES passages(passage_id)
    )
    ''')

    # Index for fast lookup during retrieval
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_passage_id ON chunks(passage_id)')
    
    conn.commit()
    conn.close()
    print(f"Database initialized successfully at {DB_PATH}")

if __name__ == "__main__":
    initialize_database()