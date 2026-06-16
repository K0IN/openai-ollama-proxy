package types

import "encoding/json"

type OllamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []OllamaMessage `json:"messages"`
	Tools     json.RawMessage `json:"tools,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Options   OllamaOptions   `json:"options,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	Think     *bool           `json:"think,omitempty"`
}

type OllamaGenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	Suffix    string          `json:"suffix,omitempty"`
	System    string          `json:"system,omitempty"`
	Template  string          `json:"template,omitempty"`
	Context   []int           `json:"context,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Raw       bool            `json:"raw,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
	Images    []string        `json:"images,omitempty"`
	Options   OllamaOptions   `json:"options,omitempty"`
	Think     *bool           `json:"think,omitempty"`
}

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
	Thinking  string           `json:"thinking,omitempty"`
}

type OllamaToolCall struct {
	Function OllamaToolCallFunction `json:"function"`
}

type OllamaToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type OllamaOptions struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	MinP             *float64 `json:"min_p,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	RepeatPenalty    *float64 `json:"repeat_penalty,omitempty"`
	NumCtx           *int     `json:"num_ctx,omitempty"`
}

type OllamaChatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            OllamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason,omitempty"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

type OllamaGenerateResponse struct {
	Model              string           `json:"model"`
	CreatedAt          string           `json:"created_at"`
	Response           string           `json:"response"`
	Thinking           string           `json:"thinking,omitempty"`
	Done               bool             `json:"done"`
	DoneReason         string           `json:"done_reason,omitempty"`
	Context            []int            `json:"context,omitempty"`
	TotalDuration      int64            `json:"total_duration,omitempty"`
	LoadDuration       int64            `json:"load_duration,omitempty"`
	PromptEvalCount    int              `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64            `json:"prompt_eval_duration,omitempty"`
	EvalCount          int              `json:"eval_count,omitempty"`
	EvalDuration       int64            `json:"eval_duration,omitempty"`
	ToolCalls          []OllamaToolCall `json:"tool_calls,omitempty"`
}

type OllamaEmbedRequest struct {
	Model      string          `json:"model"`
	Input      json.RawMessage `json:"input"`
	Prompt     string          `json:"prompt,omitempty"`
	KeepAlive  json.RawMessage `json:"keep_alive,omitempty"`
	Truncate   *bool           `json:"truncate,omitempty"`
	Dimensions int             `json:"dimensions,omitempty"`
	Options    OllamaOptions   `json:"options,omitempty"`
}

type OllamaPullRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"`
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`
	Name     string `json:"name,omitempty"`
}

type OllamaProgressResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

type OllamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

type OllamaEmbeddingsResponse struct {
	Embedding []float64 `json:"embedding"`
}

type OllamaVersionResponse struct {
	Version string `json:"version"`
}

type OllamaTagsResponse struct {
	Models []OllamaModelInfo `json:"models"`
}

type OllamaModelInfo struct {
	Name       string             `json:"name"`
	Model      string             `json:"model"`
	ModifiedAt string             `json:"modified_at"`
	Size       int64              `json:"size"`
	Digest     string             `json:"digest"`
	Details    OllamaModelDetails `json:"details"`
}

type OllamaModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type OllamaShowRequest struct {
	Model   string `json:"model"`
	Verbose bool   `json:"verbose,omitempty"`
}

type OllamaShowResponse struct {
	Modelfile    string             `json:"modelfile"`
	Parameters   string             `json:"parameters"`
	Template     string             `json:"template"`
	Details      OllamaModelDetails `json:"details"`
	ModelInfo    map[string]any     `json:"model_info"`
	Capabilities []string           `json:"capabilities"`
	License      string             `json:"license,omitempty"`
	System       string             `json:"system,omitempty"`
}

type OllamaPsResponse struct {
	Models []OllamaPsModel `json:"models"`
}

type OllamaPsModel struct {
	Name      string             `json:"name"`
	Model     string             `json:"model"`
	Size      int64              `json:"size"`
	Digest    string             `json:"digest"`
	Details   OllamaModelDetails `json:"details"`
	ExpiresAt string             `json:"expires_at"`
	SizeVRAM  int64              `json:"size_vram"`
}

type OpenAIChatRequest struct {
	Model              string                `json:"model"`
	Messages           []OpenAIMessage       `json:"messages"`
	Tools              json.RawMessage       `json:"tools,omitempty"`
	Stream             bool                  `json:"stream"`
	Temperature        *float64              `json:"temperature,omitempty"`
	TopP               *float64              `json:"top_p,omitempty"`
	Seed               *int                  `json:"seed,omitempty"`
	MaxTokens          *int                  `json:"max_tokens,omitempty"`
	Stop               []string              `json:"stop,omitempty"`
	FrequencyPenalty   *float64              `json:"frequency_penalty,omitempty"`
	PresencePenalty    *float64              `json:"presence_penalty,omitempty"`
	ResponseFormat     *OpenAIResponseFormat `json:"response_format,omitempty"`
	StreamOptions      *OpenAIStreamOptions  `json:"stream_options,omitempty"`
	TopK               *int                  `json:"top_k,omitempty"`
	MinP               *float64              `json:"min_p,omitempty"`
	RepetitionPenalty  *float64              `json:"repetition_penalty,omitempty"`
	ChatTemplateKwargs map[string]any        `json:"chat_template_kwargs,omitempty"`
}

type OpenAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type OpenAIResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type OpenAIContentPart struct {
	Type      string           `json:"type"`
	Text      string           `json:"text,omitempty"`
	ImageURL  *OpenAIImageURL  `json:"image_url,omitempty"`
	InputAudio *OpenAIAudioURL `json:"input_audio,omitempty"`
}

type OpenAIImageURL struct {
	URL string `json:"url"`
}

type OpenAIAudioURL struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type OpenAIToolCall struct {
	Index    *int                   `json:"index,omitempty"`
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIModelListResponse struct {
	Object string        `json:"object,omitempty"`
	Data   []OpenAIModel `json:"data"`
}

type OpenAIModel struct {
	Object      string `json:"object,omitempty"`
	ID          string `json:"id"`
	OwnedBy     string `json:"owned_by,omitempty"`
	Root        string `json:"root"`
	MaxModelLen int    `json:"max_model_len"`
}

type OpenAIChoice struct {
	Index        int            `json:"index"`
	Message      *OpenAIRespMsg `json:"message,omitempty"`
	Delta        *OpenAIRespMsg `json:"delta,omitempty"`
	FinishReason *string        `json:"finish_reason,omitempty"`
}

type OpenAIRespMsg struct {
	Role             string           `json:"role,omitempty"`
	Content          *string          `json:"content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
	Reasoning        *string          `json:"reasoning,omitempty"`
	ReasoningContent *string          `json:"reasoning_content,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIEmbedRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

type OpenAIEmbedResponse struct {
	Object string            `json:"object"`
	Data   []OpenAIEmbedData `json:"data"`
	Model  string            `json:"model"`
	Usage  *OpenAIUsage      `json:"usage,omitempty"`
}

type OpenAIEmbedData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}
