from typing import List, Dict


def sliding_window_chunking(
    messages: List[Dict], chunk_size: int = 5, overlap: int = 2
) -> List[List[Dict]]:

    chunks = []
    start = 0

    while start < len(messages):
        end = start + chunk_size
        chunk = messages[start:end]

        if not chunk:
            break

        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(messages):
            break

    return chunks


def build_context(chunk: List[Dict]) -> str:

    return "\n".join(f"{msg['sender']}: {msg['message']}" for msg in chunk)


def build_chat_chunks(chat_data: Dict) -> List[str]:

    messages = chat_data["messages"]

    if len(messages) <= 10:
        return [build_context(messages)]

    chunks = sliding_window_chunking(messages, chunk_size=8, overlap=3)

    text_chunks = [build_context(chunk) for chunk in chunks]

    return text_chunks
