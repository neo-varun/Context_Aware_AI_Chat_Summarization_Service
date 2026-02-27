import json
from typing import List
from chunking import build_chat_chunks
from summarizer import call_local_llm

with open("dataset\chat_messages.json", "r", encoding="utf-8") as f:
    chats = json.load(f)


def process_chats(chats: List[dict]) -> List[dict]:

    results = []

    for chat in chats:
        chat_id = chat["chat_id"]

        print(f"\nProcessing chat: {chat_id}")

        chat_chunks = build_chat_chunks(chat)

        chunk_summaries = []

        for chunk in chat_chunks:
            chunk_summary = call_local_llm(chunk)
            chunk_summaries.append(chunk_summary)

        combined_text = json.dumps(chunk_summaries, indent=2)

        final_summary = call_local_llm(
            f"""
            Combine the following summaries into one final summary.
            Remove duplicates.
            Merge similar action items.
            Return strictly in JSON format.

            {combined_text}
        """
        )

        results.append({"chat_id": chat_id, "summary": final_summary})

    return results


summaries = process_chats(chats)

print(json.dumps(summaries, indent=2))
