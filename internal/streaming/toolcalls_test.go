package streaming

import (
	"encoding/json"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestAppendToolCalls_SingleToolCall(t *testing.T) {
	states := []ToolCallState{}
	toolCalls := []types.OpenAIToolCall{
		{Function: types.OpenAIToolCallFunction{Name: "get_weather", Arguments: ""}},
	}

	result := AppendToolCalls(states, toolCalls)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	if result[0].Name != "get_weather" {
		t.Errorf("Name = %q, want %q", result[0].Name, "get_weather")
	}
}

func TestAppendToolCalls_IncrementalArguments(t *testing.T) {
	states := []ToolCallState{}

	// First delta: empty arguments
	toolCalls1 := []types.OpenAIToolCall{
		{Function: types.OpenAIToolCallFunction{Name: "get_weather", Arguments: ""}},
	}
	states = AppendToolCalls(states, toolCalls1)

	// Second delta: partial arguments "{"
	idx1 := 0
	toolCalls2 := []types.OpenAIToolCall{
		{Index: &idx1, Function: types.OpenAIToolCallFunction{Arguments: "{"}},
	}
	states = AppendToolCalls(states, toolCalls2)

	// Third delta: more arguments
	idx2 := 0
	toolCalls3 := []types.OpenAIToolCall{
		{Index: &idx2, Function: types.OpenAIToolCallFunction{Arguments: `"city":"NYC"}`}},
	}
	states = AppendToolCalls(states, toolCalls3)

	if len(states) != 1 {
		t.Fatalf("len(states) = %d, want 1", len(states))
	}
	if states[0].Name != "get_weather" {
		t.Errorf("Name = %q, want %q", states[0].Name, "get_weather")
	}
	if states[0].Arguments != `{"city":"NYC"}` {
		t.Errorf("Arguments = %q, want %q", states[0].Arguments, `{"city":"NYC"}`)
	}
}

func TestAppendToolCalls_MultipleToolCalls(t *testing.T) {
	states := []ToolCallState{}
	idx0 := 0
	idx1 := 1
	toolCalls := []types.OpenAIToolCall{
		{Index: &idx0, Function: types.OpenAIToolCallFunction{Name: "get_weather", Arguments: "{}"}},
		{Index: &idx1, Function: types.OpenAIToolCallFunction{Name: "get_time", Arguments: "{}"}},
	}

	result := AppendToolCalls(states, toolCalls)
	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
	if result[0].Name != "get_weather" {
		t.Errorf("result[0].Name = %q, want %q", result[0].Name, "get_weather")
	}
	if result[1].Name != "get_time" {
		t.Errorf("result[1].Name = %q, want %q", result[1].Name, "get_time")
	}
}

func TestAppendToolCalls_IndexBasedPlacement(t *testing.T) {
	states := []ToolCallState{{Name: "first", Arguments: "{}"}}
	idx2 := 2
	toolCalls := []types.OpenAIToolCall{
		{Index: &idx2, Function: types.OpenAIToolCallFunction{Name: "third", Arguments: "{}"}},
	}

	result := AppendToolCalls(states, toolCalls)
	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
	if result[0].Name != "first" {
		t.Errorf("result[0].Name = %q, want %q", result[0].Name, "first")
	}
	if result[2].Name != "third" {
		t.Errorf("result[2].Name = %q, want %q", result[2].Name, "third")
	}
}

func TestAppendToolCalls_EmptyToolCalls(t *testing.T) {
	states := []ToolCallState{{Name: "existing", Arguments: "{}"}}
	result := AppendToolCalls(states, nil)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	if result[0].Name != "existing" {
		t.Errorf("result[0].Name = %q, want %q", result[0].Name, "existing")
	}
}

func TestBuildOllamaToolCallChunk_ValidJSON(t *testing.T) {
	states := []ToolCallState{
		{Name: "get_weather", Arguments: `{"city":"NYC"}`},
	}

	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if !ok {
		t.Fatal("ok should be true")
	}
	if chunk.Model != "test-model" {
		t.Errorf("Model = %q, want %q", chunk.Model, "test-model")
	}
	if chunk.Done {
		t.Error("Done should be false")
	}
	if len(chunk.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(chunk.Message.ToolCalls))
	}
	if chunk.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("ToolCall name = %q, want %q", chunk.Message.ToolCalls[0].Function.Name, "get_weather")
	}
}

func TestBuildOllamaToolCallChunk_InvalidJSON(t *testing.T) {
	states := []ToolCallState{
		{Name: "broken", Arguments: `{invalid json`},
	}

	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if !ok {
		t.Fatal("ok should be true")
	}
	if len(chunk.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(chunk.Message.ToolCalls))
	}
	// Should emit empty object when JSON is invalid
	if string(chunk.Message.ToolCalls[0].Function.Arguments) != "{}" {
		t.Errorf("Arguments = %s, want {} for invalid JSON", chunk.Message.ToolCalls[0].Function.Arguments)
	}
}

func TestBuildOllamaToolCallChunk_EmptyStates(t *testing.T) {
	states := []ToolCallState{}
	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if ok {
		t.Error("ok should be false for empty states")
	}
	if len(chunk.Message.ToolCalls) != 0 {
		t.Fatalf("len(ToolCalls) = %d, want 0", len(chunk.Message.ToolCalls))
	}
}

func TestBuildOllamaToolCallChunk_EmptyNameSkipped(t *testing.T) {
	states := []ToolCallState{
		{Name: "", Arguments: "{}"},
		{Name: "valid", Arguments: "{}"},
	}

	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if !ok {
		t.Fatal("ok should be true")
	}
	if len(chunk.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1 (empty name should be skipped)", len(chunk.Message.ToolCalls))
	}
	if chunk.Message.ToolCalls[0].Function.Name != "valid" {
		t.Errorf("Name = %q, want %q", chunk.Message.ToolCalls[0].Function.Name, "valid")
	}
}

func TestBuildOllamaToolCallChunk_MultipleToolCalls(t *testing.T) {
	states := []ToolCallState{
		{Name: "get_weather", Arguments: `{"city":"NYC"}`},
		{Name: "get_time", Arguments: `{"zone":"UTC"}`},
	}

	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if !ok {
		t.Fatal("ok should be true")
	}
	if len(chunk.Message.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(chunk.Message.ToolCalls))
	}

	var args1, args2 map[string]string
	if err := json.Unmarshal(chunk.Message.ToolCalls[0].Function.Arguments, &args1); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(chunk.Message.ToolCalls[1].Function.Arguments, &args2); err != nil {
		t.Fatal(err)
	}
	if args1["city"] != "NYC" {
		t.Errorf("args1[city] = %q, want %q", args1["city"], "NYC")
	}
	if args2["zone"] != "UTC" {
		t.Errorf("args2[zone] = %q, want %q", args2["zone"], "UTC")
	}
}

func TestBuildOllamaToolCallChunk_WhitespaceOnlyArgs(t *testing.T) {
	states := []ToolCallState{
		{Name: "empty_args", Arguments: "   "},
	}

	chunk, ok := BuildOllamaToolCallChunk(states, "test-model", false)
	if !ok {
		t.Fatal("ok should be true")
	}
	if string(chunk.Message.ToolCalls[0].Function.Arguments) != "{}" {
		t.Errorf("Arguments = %s, want {} for whitespace-only args", chunk.Message.ToolCalls[0].Function.Arguments)
	}
}
