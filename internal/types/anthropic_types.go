package types

import "encoding/json"

// Anthropic Messages API types.

type AnthropicMessageRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Messages      []AnthropicMessage `json:"messages"`
	System        json.RawMessage    `json:"system,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Metadata      map[string]string  `json:"metadata,omitempty"`
	Tools         json.RawMessage    `json:"tools,omitempty"`
}

type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or []AnthropicContentBlock
}

type AnthropicContentBlock struct {
	Type string `json:"type"` // "text", "image", "tool_use", "tool_result"
	Text string `json:"text,omitempty"`

	// For tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// For tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string or []content block

	// For image
	Source *AnthropicImageSource `json:"source,omitempty"`
}

type AnthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type AnthropicMessageResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason,omitempty"`
	StopSequence *string                 `json:"stop_sequence,omitempty"`
	Usage        *AnthropicUsage         `json:"usage,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type AnthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type AnthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error AnthropicError `json:"error"`
}

// Anthropic streaming event types.

type AnthropicStreamEvent struct {
	Type string `json:"type"` // event type sent as "event: <type>\ndata: {...}\n\n"
}

type AnthropicMessageStartEvent struct {
	Type    string                   `json:"type"`
	Message AnthropicMessageResponse `json:"message"`
}

type AnthropicContentBlockStartEvent struct {
	Type         string                `json:"type"`
	Index        int                   `json:"index"`
	ContentBlock AnthropicContentBlock `json:"content_block"`
}

type AnthropicContentBlockDeltaEvent struct {
	Type  string             `json:"type"`
	Index int                `json:"index"`
	Delta AnthropicTextDelta `json:"delta"`
}

type AnthropicContentBlockDelta struct {
	Type string `json:"type"` // "text_delta", "input_json_delta"
	Text string `json:"text,omitempty"`
}

// For backward compatibility, AnthropicTextDelta is used as the delta field.
type AnthropicTextDelta struct {
	Type string `json:"type"` // "text_delta"
	Text string `json:"text,omitempty"`
}

type AnthropicContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type AnthropicMessageDeltaEvent struct {
	Type  string                `json:"type"`
	Delta AnthropicMessageDelta `json:"delta"`
	Usage *AnthropicUsage       `json:"usage,omitempty"`
}

type AnthropicMessageDelta struct {
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

type AnthropicMessageStopEvent struct {
	Type string `json:"type"`
}

type AnthropicPingEvent struct {
	Type string `json:"type"`
}
