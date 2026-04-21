package handlers

import (
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

type observedTimings struct {
	startedAt            time.Time
	responseStartedAt    time.Time
	firstVisibleOutputAt time.Time
	completedAt          time.Time
}

func newObservedTimings() *observedTimings {
	return &observedTimings{startedAt: time.Now()}
}

func (timings *observedTimings) markResponseStart() {
	if timings == nil || !timings.responseStartedAt.IsZero() {
		return
	}
	timings.responseStartedAt = time.Now()
}

func (timings *observedTimings) markFirstVisibleOutput() {
	if timings == nil || !timings.firstVisibleOutputAt.IsZero() {
		return
	}
	timings.firstVisibleOutputAt = time.Now()
}

func (timings *observedTimings) markComplete() {
	if timings == nil {
		return
	}
	timings.completedAt = time.Now()
}

func (timings *observedTimings) totalDuration() int64 {
	if timings == nil {
		return 0
	}
	return durationNanos(timings.startedAt, timings.completedAt)
}

func (timings *observedTimings) promptEvalDuration() int64 {
	if timings == nil {
		return 0
	}

	switch {
	case !timings.firstVisibleOutputAt.IsZero():
		return durationNanos(timings.startedAt, timings.firstVisibleOutputAt)
	case !timings.responseStartedAt.IsZero():
		return durationNanos(timings.startedAt, timings.responseStartedAt)
	default:
		return 0
	}
}

func (timings *observedTimings) evalDuration() int64 {
	if timings == nil {
		return 0
	}

	switch {
	case !timings.firstVisibleOutputAt.IsZero():
		return durationNanos(timings.firstVisibleOutputAt, timings.completedAt)
	case !timings.responseStartedAt.IsZero():
		return durationNanos(timings.responseStartedAt, timings.completedAt)
	default:
		return 0
	}
}

func applyObservedChatTimings(resp *types.OllamaChatResponse, timings *observedTimings) {
	if resp == nil {
		return
	}

	if total := timings.totalDuration(); total > 0 {
		resp.TotalDuration = total
	}
	if prompt := timings.promptEvalDuration(); prompt > 0 {
		resp.PromptEvalDuration = prompt
	}
	if eval := timings.evalDuration(); eval > 0 {
		resp.EvalDuration = eval
	}
}

func applyObservedGenerateTimings(resp *types.OllamaGenerateResponse, timings *observedTimings) {
	if resp == nil {
		return
	}

	if total := timings.totalDuration(); total > 0 {
		resp.TotalDuration = total
	}
	if prompt := timings.promptEvalDuration(); prompt > 0 {
		resp.PromptEvalDuration = prompt
	}
	if eval := timings.evalDuration(); eval > 0 {
		resp.EvalDuration = eval
	}
}

func applyObservedEmbedTimings(resp *types.OllamaEmbedResponse, timings *observedTimings) {
	if resp == nil {
		return
	}

	if total := timings.totalDuration(); total > 0 {
		resp.TotalDuration = total
	}
}

func durationNanos(start time.Time, end time.Time) int64 {
	if start.IsZero() || end.IsZero() || end.Before(start) {
		return 0
	}
	return end.Sub(start).Nanoseconds()
}
