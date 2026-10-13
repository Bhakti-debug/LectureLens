from gemini_client import create_gemini_client, get_model_name


client = create_gemini_client()
model = get_model_name()

response = client.models.generate_content(
    model=model,
    contents="Reply with exactly: LectureLens connection successful."
)

print(f"Model: {model}")
print(response.text)