import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gemini_client import create_gemini_client, get_model_name


def main():
    client = create_gemini_client()
    model = get_model_name()

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: LectureLens connection successful."
    )

    print(f"Model: {model}")
    print(response.text)


if __name__ == "__main__":
    main()