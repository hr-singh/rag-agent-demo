"""Interactive CLI chat with the TechNova RAG agent."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

from agent import ask, new_conversation  # noqa: E402

def main():
    print("TechNova Gadgets support agent (RAG demo). Type 'exit' to quit.\n")
    history = new_conversation()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer, history = ask(question, history)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
