import os
import json
import time
from datetime import datetime, UTC
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from chunking import build_chat_chunks
from summarizer import call_local_llm
from database import init_db, SessionLocal, ChatSummary

app = FastAPI()

init_db()

os.makedirs("output", exist_ok=True)


class ChatRequest(BaseModel):
    chat_id: str


with open("dataset/chat_messages.json", "r", encoding="utf-8") as f:
    chats = json.load(f)


def get_chat_By_id(chat_id: str) -> Dict:
    for chat in chats:
        if chat["chat_id"] == chat_id:
            return chat
    return None


@app.post("/generate_summary")
def generate_summary(request: ChatRequest):

    start_time = time.time()
    total_tokens = 0

    db = SessionLocal()

    chat = get_chat_By_id(request.chat_id)

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat_chunks = build_chat_chunks(chat)

    chunk_summaries = []

    for chunk in chat_chunks:
        result = call_local_llm(chunk)
        chunk_summaries.append(result["summary"])
        total_tokens += result["tokens"]

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

    total_tokens += result["tokens"]

    processing_time = round(time.time() - start_time, 2)

    output_file = "output/summaries.json"

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(
        {
            "chat_id": request.chat_id,
            "summary": final_summary,
            "total_tokens": total_tokens,
            "processing_time": processing_time,
        }
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    db_entry = ChatSummary(
        chat_id=request.chat_id,
        raw_chat=chat,
        generated_summary=final_summary,
        token_usage=total_tokens,
        processing_time=processing_time,
        created_at=datetime.now(UTC),
    )

    db.add(db_entry)
    db.commit()
    db.close()

    return {
        "chat_id": request.chat_id,
        "summary": final_summary,
        "token_usage": total_tokens,
        "processing_time_seconds": processing_time,
    }
