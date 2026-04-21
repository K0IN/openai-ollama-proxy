# openai-ollama-proxy

Proxy that translates Ollama API requests to OpenAI-compatible API requests. Lets Ollama clients talk to vLLM.

Works with VS Code (Ollama integration, Copilot Custom OpenAI) and the Ollama CLI — no client changes needed.

## Architecture

```
VS Code / Ollama client
        │
        ▼
┌─────────────────┐
│  ollama-proxy    │  :11434
│  (this project)  │
└────────┬────────┘
         │  OpenAI API
         ▼
┌─────────────────┐
│     vLLM         │  :8000
│  (GPU backend)   │
└─────────────────┘
```

## Run

### Locally

```bash
go build -o openai-ollama-proxy .
VLLM_BASE_URL=http://localhost:8000 VLLM_MODEL=your-model ./openai-ollama-proxy
```

### Install globally

```bash
go install github.com/k0in/openai-ollama-proxy@latest
openai-ollama-proxy
```

### Docker

```bash
# Build
docker build -t openai-ollama-proxy .

# Run
docker run -p 11434:11434 \
  -e VLLM_BASE_URL=http://host.docker.internal:8000 \
  -e VLLM_MODEL=your-model \
  openai-ollama-proxy
```

### Environment variables

| Variable | Default |
|---|---|
| `LISTEN_ADDR` | `:11434` |
| `VLLM_BASE_URL` | `http://localhost:8000` |
| `VLLM_API_KEY` | *(empty)* |
| `VLLM_MODEL` | `default` |
| `MODEL_NAME` | `qwen3:latest` |
| `MODEL_CONTEXT_LENGTH` | `65536` |
| `OLLAMA_VERSION` | `0.7.0` |
| `VLLM_STARTUP_WAIT` | `30m` |
| `VLLM_RETRY_INTERVAL` | `2s` |
| `DEBUG` | *(empty)* |

## Quickstart

```bash
docker compose up -d
```

vLLM + proxy running with Qwen 3.6. Done.

## Test with Ollama CLI

Using the proxy as ollama cli endpoint (make sure ollama is not running on the same port):

```bash
OLLAMA_HOST=http://localhost:11434 ollama run qwen3:latest "hello"
```
