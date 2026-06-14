package stats

import (
	"sync"
	"time"
)

// tokenEvent represents a single token count event with a timestamp.
type tokenEvent struct {
	timestamp    time.Time
	input        int
	output       int
	evalDuration time.Duration // time spent generating output tokens (nanos)
}

// Stats tracks token usage for the proxy.
type Stats struct {
	mu        sync.Mutex
	startTime time.Time
	// Lifetime totals
	totalInput  int
	totalOutput int
	requests    int
	// Recent events for rate calculation (sliding window)
	events []tokenEvent
	// Current request (most recent in-progress or completed)
	currentInput  int
	currentOutput int
	currentModel  string
}

// New creates a new Stats tracker.
func New() *Stats {
	return &Stats{
		startTime: time.Now(),
		events:    make([]tokenEvent, 0, 64),
	}
}

// Record records token counts from a completed request.
// evalDuration is the time spent generating output tokens (in nanoseconds),
// used to compute per-request tokens/second.
func (s *Stats) Record(model string, inputTokens, outputTokens int, evalDuration time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalInput += inputTokens
	s.totalOutput += outputTokens
	s.requests++

	// Update current to the most recent request
	s.currentInput = inputTokens
	s.currentOutput = outputTokens
	s.currentModel = model

	// Add event for rate calculation
	s.events = append(s.events, tokenEvent{
		timestamp:    time.Now(),
		input:        inputTokens,
		output:       outputTokens,
		evalDuration: evalDuration,
	})

	// Prune events older than 5 minutes to prevent unbounded growth
	cutoff := time.Now().Add(-5 * time.Minute)
	for i, e := range s.events {
		if e.timestamp.Before(cutoff) {
			if i == 0 {
				// All events are old, clear the slice but keep capacity
				s.events = s.events[:0]
				return
			}
			s.events = append(s.events[:0], s.events[i:]...)
			break
		}
	}
}

// Snapshot returns a point-in-time snapshot of the stats.
func (s *Stats) Snapshot() StatsSnapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	window := 10 * time.Second
	windowInput := 0
	windowOutput := 0
	windowRequests := 0

	for _, e := range s.events {
		if now.Sub(e.timestamp) <= window {
			windowInput += e.input
			windowOutput += e.output
			windowRequests++
		}
	}

	seconds := window.Seconds()
	inputPerSec := float64(windowInput) / seconds
	outputPerSec := float64(windowOutput) / seconds

	// Average tokens/sec across the last N requests (up to 10)
	// This gives a stable speed reading even between requests.
	const speedWindow = 10
	windowStart := len(s.events) - speedWindow
	if windowStart < 0 {
		windowStart = 0
	}
	speedSamples := s.events[windowStart:]
	var totalInputTokens int
	var totalOutputTokens int
	var totalDuration time.Duration
	for _, e := range speedSamples {
		if e.evalDuration > 0 {
			totalInputTokens += e.input
			totalOutputTokens += e.output
			totalDuration += e.evalDuration
		}
	}
	var avgInputTokensPerSec float64
	var avgOutputTokensPerSec float64
	var avgTokensPerSec float64
	if totalDuration > 0 {
		avgInputTokensPerSec = float64(totalInputTokens) / totalDuration.Seconds()
		avgOutputTokensPerSec = float64(totalOutputTokens) / totalDuration.Seconds()
		avgTokensPerSec = float64(totalInputTokens+totalOutputTokens) / totalDuration.Seconds()
	}

	return StatsSnapshot{
		Model:                 s.currentModel,
		TotalInput:            s.totalInput,
		TotalOutput:           s.totalOutput,
		Requests:              s.requests,
		Uptime:                now.Sub(s.startTime),
		CurrentInput:          s.currentInput,
		CurrentOutput:         s.currentOutput,
		InputPerSecond:        inputPerSec,
		OutputPerSecond:       outputPerSec,
		WindowRequests:        windowRequests,
		AvgInputTokensPerSec:  avgInputTokensPerSec,
		AvgOutputTokensPerSec: avgOutputTokensPerSec,
		AvgTokensPerSec:       avgTokensPerSec,
	}
}

// StatsSnapshot is an immutable snapshot of stats for external consumption.
type StatsSnapshot struct {
	Model         string
	TotalInput    int
	TotalOutput   int
	Requests      int
	Uptime        time.Duration
	CurrentInput  int
	CurrentOutput int
	// Current rates (10s sliding window)
	InputPerSecond  float64
	OutputPerSecond float64
	WindowRequests  int
	// Averages across the last 10 requests
	AvgInputTokensPerSec  float64 // average input tokens/sec across last 10 requests
	AvgOutputTokensPerSec float64 // average output tokens/sec across last 10 requests
	AvgTokensPerSec       float64 // average total tokens/sec across last 10 requests
}
