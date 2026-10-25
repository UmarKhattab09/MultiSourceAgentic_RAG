import yaml
from embeddings import ROOT, OLLAMA_URL
from openai import OpenAI

def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def query_ollama(prompt:str) -> str:
    """Query Ollama Model with a given prompt and return the response."""
    config = load_config()

    client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
    
    response_stream = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content