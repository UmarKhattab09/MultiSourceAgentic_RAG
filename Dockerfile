FROM python:3.13-slim

WORKDIR /app
# CPU torch keeps the image ~2 GB smaller; the LLM itself runs in the Ollama container
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY mcp_servers/ mcp_servers/

ENV OLLAMA_URL=http://ollama:11434
EXPOSE 8501
# bootstrap pulls missing models and builds the index on first start
CMD ["sh", "-c", "python src/bootstrap.py && streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
