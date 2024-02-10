import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Environment variable `GEMINI_API_KEY` is not set")

print(GEMINI_API_KEY)
