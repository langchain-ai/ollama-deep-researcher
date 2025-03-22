FROM --platform=$BUILDPLATFORM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (use pip for safer cross-arch install)
RUN pip install uv
ENV PATH="/root/.local/bin:${PATH}"

# 2) Copy the repository content
COPY . /app

# 3) Provide default environment variables to point to Ollama (running elsewhere)
#    Adjust the OLLAMA_URL to match your actual Ollama container or service.
ENV OLLAMA_BASE_URL="http://localhost:11434/"

# 4) Expose the port that LangGraph dev server uses (default: 2024)
EXPOSE 2024

# 5) Launch the assistant with the LangGraph dev server:
#    Equivalent to the quickstart: uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
CMD ["uvx", \
     "--refresh", \
     "--from", "langgraph-cli[inmem]", \
     "--with-editable", ".", \
     "--python", "3.11", \
     "langgraph", \
     "dev", \
     "--host", "0.0.0.0"]