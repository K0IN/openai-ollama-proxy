package stats

import (
	"sync"
	"time"
)

// tokenEvent represents a single token count event with a timestamp.
type tokenEvent struct {
	timestamp time.Time
	input     int
	output    int
}

// Stats tracks token usage for the proxy.
type Stats struct {
	mu        sync.Mutex
	startTime time.Time
	model     string
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
func (s *Stats) Record(model string, inputTokens, outputTokens int) {
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
		timestamp: time.Now(),
		input:     inputTokens,
		output:    outputTokens,
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

	return StatsSnapshot{
		Model:           s.currentModel,
		TotalInput:      s.totalInput,
		TotalOutput:     s.totalOutput,
		Requests:        s.requests,
		Uptime:          now.Sub(s.startTime),
		CurrentInput:    s.currentInput,
		CurrentOutput:   s.currentOutput,
		InputPerSecond:  inputPerSec,
		OutputPerSecond: outputPerSec,
		WindowRequests:  windowRequests,
	}
}

// StatsSnapshot is an immutable snapshot of stats for external consumption.
type StatsSnapshot struct {
	Model           string
	TotalInput      int
	TotalOutput     int
	Requests        int
	Uptime          time.Duration
	CurrentInput    int
	CurrentOutput   int
	InputPerSecond  float64
	OutputPerSecond float64
	WindowRequests  int
}
