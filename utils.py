import os


def load_system_prompt() -> str:
    base_dir = os.path.dirname((os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "prompts", "system_prompt.txt")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
