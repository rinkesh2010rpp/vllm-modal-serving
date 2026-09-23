import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from your .env file into os.environ
load_dotenv()

# 1. Look at the terminal output from your `modal deploy app.py` command
# 2. Copy the URL ending in `.modal.run` and paste it below, adding `/v1` at the end
MODAL_BASE_URL = "https://rinkesh2010rpp--llm-inference-vllmserver-serve.modal.run/v1"
client = OpenAI(
    base_url=MODAL_BASE_URL,
    api_key="dummy",  # web endpoint is unauthenticated; set --api-key in index.py to require one
)

print("Sending request to Qwen on Modal...")

# Non-streaming call
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "You are a helpful and direct AI assistant."},
        {"role": "user", "content": "Explain what an API is in one simple sentence."},
    ],
    temperature=0.7,
)

print("\nResponse:")
print(response.choices[0].message.content)


# Streaming call
print("\nTesting streaming response:")
stream = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Count from 1 to 5, one per line."},
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()