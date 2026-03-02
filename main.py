import json
import time
from datetime import datetime,UTC
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from chunking import build_chat_chunks
from summarizer import call_local_llm
from database import init_db, SessionLocal, ChatSummary

init_db()

app=FastAPI()

class ChatRequest(BaseModel):
    chat_id: str

with open("dataset/chat_messages.json", "r", encoding="utf-8") as f:
    chats = json.load(f)

def get_chat_By_id(chat_id: str) -> Dict:
    for chat in chats:
        if chat['chat_id']==chat_id:
            return chat
    return None

@app.post('/generate_summary')
def generate_summary(request: ChatRequest):

    start_time=time.time()

    db=SessionLocal()

    chat=get_chat_By_id(request.chat_id)

    if not chat:
        raise HTTPException(status_code=404, detail='Chat not found')

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

    processing_time=round(time.time()-start_time,2)

    db_entry=ChatSummary(
        chat_id=request.chat_id,
        raw_chat=chat,
        generated_summary=final_summary,
        token_usage=0,
        processing_time=processing_time,
        created_at=datetime.now(UTC)
    )

    db.add(db_entry)
    db.commit()
    db.close()

    return {
        'chat_id': request.chat_id,
        'summary': final_summary,
        'processing_time_seconds':processing_time
    }