import sys
import warnings

from config import DEEPINFRA_API_KEY
from rag import answer_question, format_context, retrieve_relevant_chunks


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=DeprecationWarning)


QUESTIONS = [
    "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
]


def main() -> None:
    for question in QUESTIONS:
        print("=" * 100)
        print(f"Question: {question}")
        if DEEPINFRA_API_KEY:
            result = answer_question(question)
            print(f"Latency: {result.latency_seconds:.2f} seconds")
            print(f"Answer: {result.answer}")
            print("Sources:")
            print(format_context(result.sources))
        else:
            print("LLM call skipped: DEEPINFRA_API_KEY is missing.")
            print("Retrieved sources:")
            print(format_context(retrieve_relevant_chunks(question)))


if __name__ == "__main__":
    main()
