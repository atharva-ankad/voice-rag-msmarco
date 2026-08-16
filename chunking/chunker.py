import tiktoken
import re

# Use cl100k_base tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def split_into_sentences(text: str) -> list[str]:
    """Lightweight regex-based multilingual sentence splitter for Indic and Latin scripts."""
    # Matches Latin (.!?), Devanagari Danda (।), Double Danda (॥), and newlines
    sentence_endings = re.compile(r'(?<=[.!?।॥\n])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_semantic(text: str, max_tokens: int = 300) -> list[str]:
    """Strategy A: Pack sentences sequentially up to max_tokens with 1-sentence overlap."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        sent_tokens = count_tokens(sent)
        
        # If single sentence exceeds max_tokens, split it using fixed window fallback
        if sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.extend(chunk_fixed_sliding(sent, window_size=max_tokens, overlap=50))
            continue

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # 1-sentence overlap
            overlap_sent = current_chunk[-1]
            current_chunk = [overlap_sent, sent]
            current_tokens = count_tokens(overlap_sent) + sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_fixed_sliding(text: str, window_size: int, overlap: int) -> list[str]:
    """Strategy B: Fixed token sliding window with token-level overlap."""
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    chunks = []
    step = window_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + window_size]
        chunk_text = tokenizer.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if i + window_size >= len(tokens):
            break

    return chunks