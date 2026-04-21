package streaming

import (
	"encoding/json"
	"log"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

type ToolCallState struct {
	Name      string
	Arguments string
}

func AppendToolCalls(states []ToolCallState, toolCalls []types.OpenAIToolCall) []ToolCallState {
	for idx, toolCall := range toolCalls {
		stateIndex := idx
		if toolCall.Index != nil && *toolCall.Index >= 0 {
			stateIndex = *toolCall.Index
		}

		for len(states) <= stateIndex {
			states = append(states, ToolCallState{})
		}

		if toolCall.Function.Name != "" {
			states[stateIndex].Name += toolCall.Function.Name
		}
		if toolCall.Function.Arguments != "" {
			states[stateIndex].Arguments += toolCall.Function.Arguments
		}
	}

	return states
}

func BuildOllamaToolCallChunk(states []ToolCallState, model string, debug bool) (types.OllamaChatResponse, bool) {
	chunk := types.OllamaChatResponse{
		Model:     model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Done:      false,
		Message:   types.OllamaMessage{Role: "assistant"},
	}

	for _, state := range states {
		if state.Name == "" {
			continue
		}

		args := strings.TrimSpace(state.Arguments)
		rawArgs := json.RawMessage("{}")
		if args != "" {
			if json.Valid([]byte(args)) {
				rawArgs = json.RawMessage(args)
			} else if debug {
				log.Printf("stream tool call invalid JSON for %q, emitting empty object", state.Name)
			}
		}

		chunk.Message.ToolCalls = append(chunk.Message.ToolCalls, types.OllamaToolCall{
			Function: types.OllamaToolCallFunction{
				Name:      state.Name,
				Arguments: rawArgs,
			},
		})
	}

	return chunk, len(chunk.Message.ToolCalls) > 0
}
