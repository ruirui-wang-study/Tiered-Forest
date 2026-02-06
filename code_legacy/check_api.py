import os
import configparser
from openai import OpenAI
import httpx

config = configparser.ConfigParser()
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(base_dir), 'config.ini')
config.read(config_path)

api_key = config.get('api', 'deepseek_key')
base_url = config.get('api', 'deepseek_url')

print(f"Checking API: {base_url}")
try:
    # Try with default verification
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Standard Client Failed:", e)
    print("Trying verify=False...")
    try:
        # Try disable SSL verify
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("Success with verify=False:", response.choices[0].message.content)
    except Exception as e2:
        print("verify=False Failed:", e2)
