import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Environment variable `GEMINI_API_KEY` is not set")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("What is Plaksha University?")

print(response.text)
