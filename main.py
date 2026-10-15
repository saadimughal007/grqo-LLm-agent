
"""A simple command-line AI assistant using Groq's API."""

import os
import sys

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# Load configuration from the local .env file
load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "").strip()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))

INSTRUCTIONS = (
    "You are a helpful assistant. Answer clearly and keep the response "
    "reasonably short unless the user asks for more detail."
)


def ask(client: Groq, question: str):
    """Send one question to Groq and return the API response."""
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "PASTE_YOUR_PRIVATE_GROQ_KEY_HERE":
        sys.exit("GROQ_API_KEY missing. Add your private key to .env.")

    if not MODEL:
        sys.exit("GROQ_MODEL missing. Add a supported model name to .env.")

    if MAX_OUTPUT_TOKENS < 1:
        sys.exit("MAX_OUTPUT_TOKENS must be a positive integer.")

    client = Groq(api_key=api_key)

    print(f"Groq AI assistant | Model: {MODEL}")
    print("Enter a question. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        try:
            response = ask(client, question)
            answer = response.choices[0].message.content or ""
            print(f"\nAI: {answer}\n")

            if response.usage:
                print(
                    f"[Tokens - input: {response.usage.prompt_tokens}, "
                    f"output: {response.usage.completion_tokens}, "
                    f"total: {response.usage.total_tokens}]\n"
                )

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(
                    "Note: The response may have been cut off by the "
                    "output-token limit.\n"
                )

        except AuthenticationError:
            print("Authentication failed. Check your Groq API key.\n")
        except RateLimitError:
            print("Rate limit or quota reached. Check your Groq account.\n")
        except APIConnectionError:
            print("Connection failed. Check your internet connection.\n")
        except APIStatusError as error:
            print(f"Groq API error ({error.status_code}): {error.message}\n")


if __name__ == "__main__":
    main()
