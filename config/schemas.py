from pydantic import BaseModel
from typing import Optional

class PassageRecord(BaseModel):
    passage_id: str
    query_id: str
    source_language: str
    target_language: str
    english_text: str
    translated_text: str
    is_selected: int
    split: str

class ChunkRecord(BaseModel):
    chunk_id: str
    passage_id: str
    strategy: str
    text: str
    token_count: int