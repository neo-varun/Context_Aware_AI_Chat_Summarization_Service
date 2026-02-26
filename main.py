import json
from chunking import process_chat

with open("dataset/chat_messages.json", "r", encoding="utf-8") as f:
    chats = json.load(f)

for chat in chats:
    print(f"\nProcessing chat: {chat['chat_id']}")

    summary = process_chat(chat)

    print(f"Final Summary: {summary}")
