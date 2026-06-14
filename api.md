# API Reference

## Table of Contents
- [Ollama API Endpoints](#ollama-api-endpoints)
  - [Chat Completions](#chat-completions)
  - [Generate Completions](#generate-completions)
  - [Embeddings](#embeddings)
  - [Model Management](#model-management)
- [OpenAI API Endpoints](#openai-api-endpoints)
  - [Models](#models)
  - [Chat Completions](#chat-completions-1)
  - [Embeddings](#embeddings-1)
- [Anthropic API Endpoints](#anthropic-api-endpoints)
  - [Messages](#messages)
- [Proxy Endpoints](#proxy-endpoints)
  - [Stats](#stats)
  - [Health](#health)

---

## Ollama API Endpoints

Ollama-compatible endpoints for clients that speak the Ollama protocol.

### Chat Completions

**`POST /api/chat`**

Send a chat completion request in Ollama format.

**Request Body:**
```json
{
  "model": "string",
  "messages": [
    {
      "role": "system" | "user" | "assistant",
      "content": "string"
    }
  ],
  "stream": boolean (optional, default: true),
  "options": {
    "temperature": number,
    "top_p": number,
    "num_predict": number
  }
}
```

**Response (non-streaming):**
```json
{
  "model": "string",
  "created_at": "2024-01-01T00:00:00Z",
  "message": {
    "role": "assistant",
    "content": "string"
  },
  "done": true,
  "total_duration": 123456789,
  "load_duration": 12345678,
  "prompt_eval_count": 10,
  "prompt_eval_duration": 12345678,
  "eval_count": 50,
  "eval_duration": 123456789
}
```

**Response (streaming):**
Returns NDJSON stream of `OllamaChatResponse` objects.

---

### Generate Completions

**`POST /api/generate`**

Send a text generation request in Ollama format.

**Request Body:**
```json
{
  "model": "string",
  "prompt": "string",
  "stream": boolean (optional),
  "options": {
    "temperature": number,
    "top_p": number,
    "num_predict": number
  }
}
```

**Response (non-streaming):**
```json
{
  "model": "string",
  "created_at": "2024-01-01T00:00:00Z",
  "response": "string",
  "done": true,
  "total_duration": 123456789,
  "load_duration": 12345678,
  "prompt_eval_count": 10,
  "prompt_eval_duration": 12345678,
  "eval_count": 50,
  "eval_duration": 123456789
}
```

**Response (streaming):**
Returns NDJSON stream of `OllamaGenerateResponse` objects.

---

### Embeddings

**`POST /api/embed`**

Generate embeddings for input text.

**Request Body:**
```json
{
  "model": "string",
  "input": "string",
  "truncate": boolean (optional)
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

---

### Model Management

#### List Tags

**`GET /api/tags`**

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "string",
      "model": "string",
      "modified_at": "2024-01-01T00:00:00Z",
      "size": 0,
      "digest": "proxy",
      "details": {
        "parent_model": "string",
        "format": "string",
        "family": "string",
        "parameter_size": "string",
        "quantization_level": "string"
      }
    }
  ]
}
```

#### Show Model Info

**`POST /api/show`**

Get detailed information about the current model.

**Response:**
```json
{
  "modelfile": "# proxied model",
  "parameters": "num_ctx 4096",
  "template": "",
  "details": { ... },
  "model_info": { ... },
  "capabilities": ["completion", "tools"]
}
```

#### Version

**`GET /api/version`**

Get the Ollama version string.

**Response:**
```json
{
  "version": "0.x.x"
}
```

#### Process List

**`GET /api/ps`**

List loaded models (shows the proxied model).

**Response:**
```json
{
  "models": [
    {
      "name": "string",
      "model": "string",
      "size": 0,
      "digest": "proxy",
      "details": { ... },
      "expires_at": "2024-01-02T00:00:00Z",
      "size_vram": 0
    }
  ]
}
```

#### Pull, Push, Create, Copy, Delete

Stub endpoints that return success responses:
- **`POST /api/pull`** - Simulates pulling a model
- **`POST /api/push`** - Simulates pushing a model
- **`POST /api/create`** - Simulates creating a model
- **`POST /api/copy`** - Simulates copying a model
- **`DELETE /api/delete`** - Simulates deleting a model
- **`HEAD /api/blobs/{hash}`** - Check if blob exists
- **`POST /api/blobs/{hash}`** - Upload blob

---

## OpenAI API Endpoints

OpenAI-compatible endpoints for clients that speak the OpenAI protocol.

### Models

**`GET /models`** or **`GET /v1/models`**

List available models (OpenAI format).

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "model",
      "id": "string",
      "owned_by": "openai-ollama-proxy",
      "root": "string",
      "max_model_len": 4096
    }
  ]
}
```

---

### Chat Completions

**`POST /chat/completions`** or **`POST /v1/chat/completions`**

Send a chat completion request in OpenAI format.

**Request Body:**
```json
{
  "model": "string",
  "messages": [
    {
      "role": "system" | "user" | "assistant",
      "content": "string"
    }
  ],
  "stream": boolean (optional),
  "temperature": number,
  "top_p": number,
  "max_tokens": number,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "string",
        "description": "string",
        "parameters": { ... }
      }
    }
  ]
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "string",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "string"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**Response (streaming):**
Returns SSE stream of `OpenAIChatResponse` chunks.

---

### Embeddings

**`POST /embeddings`** or **`POST /v1/embeddings`**

Generate embeddings for input text (OpenAI format).

**Request Body:**
```json
{
  "model": "string",
  "input": "string" | ["string", "string"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 10
  }
}
```

---

## Anthropic API Endpoints

Anthropic-compatible endpoints for clients that speak the Anthropic Messages API format (e.g., Claude Code CLI).

### Messages

**`POST /messages`** or **`POST /v1/messages`**

Send a messages request in Anthropic format. The proxy translates it to an OpenAI chat completion request, forwards it upstream, and translates the response back.

**Request Body:**
```json
{
  "model": "string",
  "max_tokens": 8192,
  "messages": [
    {
      "role": "user" | "assistant",
      "content": "Hello!"
    }
  ],
  "system": "You are a helpful assistant.",
  "stream": boolean (optional, default: false),
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["\\n\\n"],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the weather",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        }
      }
    }
  ]
}
```

**Response (non-streaming):**
```json
{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! How can I help you today?"
    }
  ],
  "model": "string",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 50
  }
}
```

**Response (streaming):**
Returns SSE stream of Anthropic events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`).

**Error Response:**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "description of the error"
  }
}
```

---

## Proxy Endpoints

Proxy-specific endpoints for monitoring and health checks.

### Stats

**`GET /stats`**

Get real-time statistics about the proxy. Returns JSON with lifetime totals, current request info, and token rates.

**Response:**
```json
{
  "model": "qwen3-27b",
  "stats": {
    "total_input_tokens": 12345,
    "total_output_tokens": 67890,
    "total_tokens": 80235,
    "total_requests": 42,
    "uptime_seconds": 3600.5,
    "current_input_tokens": 100,
    "current_output_tokens": 500,
    "input_tokens_per_sec": 15.5,
    "output_tokens_per_sec": 45.2,
    "tokens_per_sec": 60.7,
    "avg_input_tokens_per_sec": 12.3,
    "avg_output_tokens_per_sec": 36.0,
    "avg_tokens_per_sec": 48.3
  }
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Name of the most recent model used |
| `total_input_tokens` | int | Total input tokens processed since proxy start |
| `total_output_tokens` | int | Total output tokens generated since proxy start |
| `total_tokens` | int | Sum of input and output tokens |
| `total_requests` | int | Total number of requests processed |
| `uptime_seconds` | float | Seconds since proxy started |
| `current_input_tokens` | int | Input tokens from the most recent request |
| `current_output_tokens` | int | Output tokens from the most recent request |
| `input_tokens_per_sec` | float | Input token rate (10s sliding window) |
| `output_tokens_per_sec` | float | Output token rate (10s sliding window) |
| `tokens_per_sec` | float | Combined token rate (10s sliding window) |
| `avg_input_tokens_per_sec` | float | Average input tokens/sec across the last 10 requests |
| `avg_output_tokens_per_sec` | float | Average output tokens/sec across the last 10 requests |
| `avg_tokens_per_sec` | float | Average total tokens/sec across the last 10 requests |

**Example usage (waybar):**
```bash
# Query output tokens/sec (like a network widget)
curl -s http://localhost:11434/stats | jq -r '.stats.output_tokens_per_sec'

# Query total tokens/sec (input + output)
curl -s http://localhost:11434/stats | jq -r '.stats.tokens_per_sec'

# Format for waybar display
curl -s http://localhost:11434/stats | jq -r '"\(.stats.output_tokens_per_sec | round) tok/s"'
```

---

### Health

**`GET /`**

Check if the proxy and upstream are healthy.

**Response:**
- `200 OK` - "Ollama is running" (upstream is healthy)
- `503 Service Unavailable` - "Ollama is down" (upstream is unhealthy)

**`HEAD /`**

Same as GET but without response body (for health checks).

---

## Error Responses

All endpoints may return:
- `400 Bad Request` - Invalid request body or parameters
- `405 Method Not Allowed` - Wrong HTTP method
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Upstream is not available
