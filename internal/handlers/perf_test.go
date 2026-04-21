package handlers

import (
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestApplyObservedChatTimings_UsesVisibleOutputBoundaries(t *testing.T) {
	timings := &observedTimings{
		startedAt:            time.Unix(0, 0),
		responseStartedAt:    time.Unix(0, 1*time.Millisecond.Nanoseconds()),
		firstVisibleOutputAt: time.Unix(0, 3*time.Millisecond.Nanoseconds()),
		completedAt:          time.Unix(0, 10*time.Millisecond.Nanoseconds()),
	}

	resp := types.OllamaChatResponse{}
	applyObservedChatTimings(&resp, timings)

	if resp.TotalDuration != int64(10*time.Millisecond) {
		t.Fatalf("TotalDuration = %d, want %d", resp.TotalDuration, int64(10*time.Millisecond))
	}
	if resp.PromptEvalDuration != int64(3*time.Millisecond) {
		t.Fatalf("PromptEvalDuration = %d, want %d", resp.PromptEvalDuration, int64(3*time.Millisecond))
	}
	if resp.EvalDuration != int64(7*time.Millisecond) {
		t.Fatalf("EvalDuration = %d, want %d", resp.EvalDuration, int64(7*time.Millisecond))
	}
	if resp.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0", resp.LoadDuration)
	}
}

func TestApplyObservedChatTimings_FallsBackToResponseStart(t *testing.T) {
	timings := &observedTimings{
		startedAt:         time.Unix(0, 0),
		responseStartedAt: time.Unix(0, 4*time.Millisecond.Nanoseconds()),
		completedAt:       time.Unix(0, 6*time.Millisecond.Nanoseconds()),
	}

	resp := types.OllamaChatResponse{}
	applyObservedChatTimings(&resp, timings)

	if resp.TotalDuration != int64(6*time.Millisecond) {
		t.Fatalf("TotalDuration = %d, want %d", resp.TotalDuration, int64(6*time.Millisecond))
	}
	if resp.PromptEvalDuration != int64(4*time.Millisecond) {
		t.Fatalf("PromptEvalDuration = %d, want %d", resp.PromptEvalDuration, int64(4*time.Millisecond))
	}
	if resp.EvalDuration != int64(2*time.Millisecond) {
		t.Fatalf("EvalDuration = %d, want %d", resp.EvalDuration, int64(2*time.Millisecond))
	}
}

func TestApplyObservedEmbedTimings_UsesMeasuredTotalOnly(t *testing.T) {
	timings := &observedTimings{
		startedAt:         time.Unix(0, 0),
		completedAt:       time.Unix(0, 5*time.Millisecond.Nanoseconds()),
		responseStartedAt: time.Unix(0, 2*time.Millisecond.Nanoseconds()),
	}

	resp := types.OllamaEmbedResponse{}
	applyObservedEmbedTimings(&resp, timings)

	if resp.TotalDuration != int64(5*time.Millisecond) {
		t.Fatalf("TotalDuration = %d, want %d", resp.TotalDuration, int64(5*time.Millisecond))
	}
	if resp.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0", resp.LoadDuration)
	}
}
