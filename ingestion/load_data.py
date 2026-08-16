import sqlite3
import os
from datasets import load_dataset

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag_sidecar.db')
DEV_RECORD_LIMIT = 1000 # Strict boundary for local development and API testing

def ingest_data():
    print(f"Connecting to database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Streaming MSMARCO-XI (Hindi) via PR #2...")
    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI",
        "hi",
        split="validation",
        streaming=True,
        revision="refs/pr/2"
    )

    inserted_queries = 0
    inserted_passages = 0

    for row in dataset:
        if inserted_queries >= DEV_RECORD_LIMIT:
            break

        query_id = str(row['query_id'])
        source_lang = row.get('source_lang', 'eng_Latn')
        target_lang = row.get('target_lang', 'hin_Deva')
        
        passages = row['passages']
        is_selected_list = passages['is_selected']
        eng_passages = passages['English_passages']
        trans_passages = passages['Translated_passages']

        # Guardrail: Drop queries lacking positive retrievable evidence
        if sum(is_selected_list) == 0:
            continue

        # Flatten nested passages into relational rows
        for i in range(len(is_selected_list)):
            passage_id = f"{query_id}_p{i}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO passages 
                (passage_id, query_id, source_language, target_language, english_text, translated_text, is_selected, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                passage_id,
                query_id,
                source_lang,
                target_lang,
                eng_passages[i],
                trans_passages[i],
                is_selected_list[i],
                'validation'
            ))
            inserted_passages += 1
        
        inserted_queries += 1
        
        if inserted_queries % 100 == 0:
            print(f"Processed {inserted_queries} valid queries. Passages flattened: {inserted_passages}...")
            conn.commit()

    conn.commit()
    conn.close()
    
    print("\n--- Phase 2 Ingestion Complete ---")
    print(f"Total Valid Queries: {inserted_queries}")
    print(f"Total Passages Inserted: {inserted_passages}")

if __name__ == "__main__":
    ingest_data()