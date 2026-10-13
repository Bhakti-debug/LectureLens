from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL


def create_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def get_model_name():
    return GEMINI_MODEL