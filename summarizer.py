from ollama import chat
from ollama import ChatResponse


def call_local_llm(content):
    response: ChatResponse = chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    return response.message.content
