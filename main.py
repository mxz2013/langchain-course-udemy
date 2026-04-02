import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")
    print(
        os.environ.get("OPENAI_API_KEY")
    )  # Example of accessing an environment variable


if __name__ == "__main__":
    main()
