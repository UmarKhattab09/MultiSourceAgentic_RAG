from llm import query_ollama

prompt = "What is the capital of France?"

for response in query_ollama(prompt):
    print(response, end="", flush=True)
