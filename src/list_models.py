"""Print the Groq models your API key can use:  python -m src.list_models"""
from dotenv import load_dotenv

load_dotenv()
from src.llm import list_available_models, pick_model  # noqa: E402

if __name__ == "__main__":
    for m in list_available_models():
        print(m)
    print("\nAuto-selected:", pick_model())
