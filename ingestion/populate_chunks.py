import sqlite3
import os
import sys

# Ensure root directory is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking.chunker import chunk_semantic, chunk_fixed_sliding, count_tokens

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag_sidecar.db')

def populate_all_chunks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"Reading passages from {DB_PATH}...")
    cursor.execute("SELECT passage_id, english_text, translated_text FROM passages")
    rows = cursor.fetchall()
    print(f"Total passages to chunk: {len(rows)}")

    chunk_records = []

    for passage_id, eng_text, trans_text in rows:
        # We index both English and Translated text
        targets = [
            ("en", eng_text or ""),
            ("hi", trans_text or "")
        ]

        for lang, text in targets:
            if not text.strip():
                continue

            # Strategy A: Semantic (300 tokens, 1 sentence overlap)
            sem_chunks = chunk_semantic(text, max_tokens=300)
            for idx, c_text in enumerate(sem_chunks):
                chunk_id = f"{passage_id}_{lang}_sem_{idx}"
                chunk_records.append((chunk_id, passage_id, "semantic_300", c_text, count_tokens(c_text)))

            # Strategy B1: Fixed (256 window, 50 overlap)
            f256_chunks = chunk_fixed_sliding(text, window_size=256, overlap=50)
            for idx, c_text in enumerate(f256_chunks):
                chunk_id = f"{passage_id}_{lang}_f256_{idx}"
                chunk_records.append((chunk_id, passage_id, "fixed_256_50", c_text, count_tokens(c_text)))

            # Strategy B2: Fixed (512 window, 100 overlap)
            f512_chunks = chunk_fixed_sliding(text, window_size=512, overlap=100)
            for idx, c_text in enumerate(f512_chunks):
                chunk_id = f"{passage_id}_{lang}_f512_{idx}"
                chunk_records.append((chunk_id, passage_id, "fixed_512_100", c_text, count_tokens(c_text)))

    print(f"Inserting {len(chunk_records)} total chunks into SQLite...")
    cursor.executemany('''
        INSERT OR REPLACE INTO chunks (chunk_id, passage_id, strategy, text, token_count)
        VALUES (?, ?, ?, ?, ?)
    ''', chunk_records)

    conn.commit()
    conn.close()
    print(f"Successfully populated {len(chunk_records)} chunks across 3 strategies.")

if __name__ == "__main__":
    populate_all_chunks()