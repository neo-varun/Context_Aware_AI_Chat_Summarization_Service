from typing import List, Dict
from summarizer import call_local_llm


def sliding_window_chunking(
    messages: List[Dict], chunk_size: int = 5, overlap: int = 2
) -> List[List[Dict]]:

    chunks = []
    start = 0

    while start < len(messages):
        end = start + chunk_size
        chunk = messages[start:end]
        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(messages):
            break

    return chunks


def build_context(chunk: List[Dict]) -> str:

    return "\n".join(f"{msg['sender']}: {msg['message']}" for msg in chunk)


def join_summaries(chunk_summaries: List[str]) -> str:
    combined = "\n\n".join(chunk_summaries)

    prompt = f"""
    Combine the following partial summaries into one coherent summary.
    Remove duplicates.
    Merge similar action items.

    {combined}
    """

    return call_local_llm(prompt)


def process_chat(chat_data):

    messages = chat_data["messages"]

    if len(messages) <= 10:
        context = build_context(messages)
        return call_local_llm(context)

    chunks = sliding_window_chunking(messages, chunk_size=8, overlap=3)

    chunk_summaries = []

    for chunk in chunks:
        context = build_context(chunk)
        summary = call_local_llm(context)
        chunk_summaries.append(summary)

    final_summary = join_summaries(chunk_summaries)

    return final_summary
