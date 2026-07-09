package stats

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

type tokenEvent struct {
	timestamp    time.Time
	input        int
	output       int
	evalDuration time.Duration // time spent generating output tokens (nanos)
	model        string
}

type modelStat struct {
	totalInput  int
	totalOutput int
	requests    int
	totalEval   time.Duration
}

// dailyModelStat tracks token usage for a specific model on a specific day.
type dailyModelStat struct {
	input    int
	output   int
	requests int
}

type Stats struct {
	mu            sync.Mutex
	startTime     time.Time
	totalInput    int
	totalOutput   int
	requests      int
	events        []tokenEvent
	currentInput  int
	currentOutput int
	currentModel  string
	perModel      map[string]*modelStat
	// dailyStats is keyed by "YYYY-MM-DD/model" for per-day per-model tracking.
	dailyStats map[string]*dailyModelStat
}

func New() *Stats {
	return &Stats{
		startTime:  time.Now(),
		events:     make([]tokenEvent, 0, 64),
		perModel:   make(map[string]*modelStat),
		dailyStats: make(map[string]*dailyModelStat),
	}
}

func (s *Stats) Record(model string, inputTokens, outputTokens int, evalDuration time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalInput += inputTokens
	s.totalOutput += outputTokens
	s.requests++

	s.currentInput = inputTokens
	s.currentOutput = outputTokens
	s.currentModel = model

	// Track per-model stats
	ms, ok := s.perModel[model]
	if !ok {
		ms = &modelStat{}
		s.perModel[model] = ms
	}
	ms.totalInput += inputTokens
	ms.totalOutput += outputTokens
	ms.requests++
	ms.totalEval += evalDuration

	s.events = append(s.events, tokenEvent{
		timestamp:    time.Now(),
		input:        inputTokens,
		output:       outputTokens,
		evalDuration: evalDuration,
		model:        model,
	})

	// Track daily per-model stats (keyed by "YYYY-MM-DD/model")
	dayKey := time.Now().Format("2006-01-02") + "/" + model
	ds, ok := s.dailyStats[dayKey]
	if !ok {
		ds = &dailyModelStat{}
		s.dailyStats[dayKey] = ds
	}
	ds.input += inputTokens
	ds.output += outputTokens
	ds.requests++

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

	// Build per-model snapshot
	perModel := make(map[string]PerModelStats, len(s.perModel))
	for model, ms := range s.perModel {
		var outputTokensPerSec float64
		if ms.totalEval > 0 {
			outputTokensPerSec = float64(ms.totalOutput) / ms.totalEval.Seconds()
		}
		perModel[model] = PerModelStats{
			TotalInput:         ms.totalInput,
			TotalOutput:        ms.totalOutput,
			TotalTokens:        ms.totalInput + ms.totalOutput,
			Requests:           ms.requests,
			OutputTokensPerSec: outputTokensPerSec,
		}
	}

	// Build daily per-model snapshot for the last 7 days
	daily := buildDailyStats(s.dailyStats)

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
		PerModel:              perModel,
		Daily:                 daily,
	}
}

type PerModelStats struct {
	TotalInput         int
	TotalOutput        int
	TotalTokens        int
	Requests           int
	OutputTokensPerSec float64
}

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
	// Per-model stats (overall)
	PerModel map[string]PerModelStats
	// Daily per-model breakdown for the last 7 days
	Daily map[string]map[string]DailyModelStats
}

// DailyModelStats holds token usage for a single model on a single day.
type DailyModelStats struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
	Requests     int `json:"requests"`
}

// --- Persistence ------------------------------------------------------------

// saveData is the JSON-serializable subset of Stats for disk persistence.
type saveData struct {
	TotalInput    int                        `json:"total_input_tokens"`
	TotalOutput   int                        `json:"total_output_tokens"`
	Requests      int                        `json:"total_requests"`
	CurrentModel  string                     `json:"current_model"`
	CurrentInput  int                        `json:"current_input_tokens"`
	CurrentOutput int                        `json:"current_output_tokens"`
	PerModel      map[string]*modelStat      `json:"per_model"`
	DailyStats    map[string]*dailyModelStat `json:"daily_stats,omitempty"`
}

// modelStat for JSON serialization
type modelStatJSON struct {
	TotalInput  int   `json:"total_input_tokens"`
	TotalOutput int   `json:"total_output_tokens"`
	Requests    int   `json:"total_requests"`
	TotalEvalNs int64 `json:"total_eval_ns"`
}

// Save persists the current aggregated stats to a JSON file at path.
func (s *Stats) Save(path string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := saveData{
		TotalInput:    s.totalInput,
		TotalOutput:   s.totalOutput,
		Requests:      s.requests,
		CurrentModel:  s.currentModel,
		CurrentInput:  s.currentInput,
		CurrentOutput: s.currentOutput,
		PerModel:      s.perModel,
		DailyStats:    s.dailyStats,
	}

	payload, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(path, payload, 0600)
}

// LoadFromFile restores aggregated stats from a JSON file previously written
// by Save. The returned Stats already has the accumulated values; events are
// empty (sliding-window rates start fresh after restart).
// If path is empty or the file does not exist, a fresh Stats is returned.
func LoadFromFile(path string) (*Stats, error) {
	if path == "" {
		return New(), nil
	}
	data, err := os.ReadFile(path) // #nosec G304 -- path comes from user config, not arbitrary input
	if err != nil {
		if os.IsNotExist(err) {
			return New(), nil
		}
		return nil, err
	}

	var raw saveData
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	s := New()
	s.totalInput = raw.TotalInput
	s.totalOutput = raw.TotalOutput
	s.requests = raw.Requests
	s.currentModel = raw.CurrentModel
	s.currentInput = raw.CurrentInput
	s.currentOutput = raw.CurrentOutput
	s.perModel = raw.PerModel
	if s.perModel == nil {
		s.perModel = make(map[string]*modelStat)
	}
	s.dailyStats = raw.DailyStats
	if s.dailyStats == nil {
		s.dailyStats = make(map[string]*dailyModelStat)
	}

	return s, nil
}

// MarshalJSON implements json.Marshaler for modelStat.
func (m *modelStat) MarshalJSON() ([]byte, error) {
	return json.Marshal(modelStatJSON{
		TotalInput:  m.totalInput,
		TotalOutput: m.totalOutput,
		Requests:    m.requests,
		TotalEvalNs: int64(m.totalEval),
	})
}

// UnmarshalJSON implements json.Unmarshaler for modelStat.
func (m *modelStat) UnmarshalJSON(data []byte) error {
	var j modelStatJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	m.totalInput = j.TotalInput
	m.totalOutput = j.TotalOutput
	m.requests = j.Requests
	m.totalEval = time.Duration(j.TotalEvalNs)
	return nil
}

// --- Daily stats helpers ----------------------------------------------------

// dailyModelStatJSON is the serialization format for daily stats.
type dailyModelStatJSON struct {
	Input    int `json:"input_tokens"`
	Output   int `json:"output_tokens"`
	Requests int `json:"requests"`
}

// buildDailyStats builds a map of date -> model -> DailyModelStats for the
// last 7 days. It prunes entries older than 7 days to prevent unbounded growth.
func buildDailyStats(dailyStats map[string]*dailyModelStat) map[string]map[string]DailyModelStats {
	if len(dailyStats) == 0 {
		return nil
	}

	now := time.Now()
	cutoffDate := now.AddDate(0, 0, -6).Format("2006-01-02")

	// Collect all unique dates and prune old entries.
	dateSet := make(map[string]bool)
	for key := range dailyStats {
		// Key format: "YYYY-MM-DD/model"
		slashIdx := stringsIndex(key, "/")
		if slashIdx <= 0 {
			continue
		}
		dateStr := key[:slashIdx]
		// Prune entries older than 7 days.
		if dateStr < cutoffDate {
			delete(dailyStats, key)
			continue
		}
		dateSet[dateStr] = true
	}

	result := make(map[string]map[string]DailyModelStats)
	for dateStr := range dateSet {
		result[dateStr] = make(map[string]DailyModelStats)
	}

	for key, ds := range dailyStats {
		slashIdx := stringsIndex(key, "/")
		if slashIdx <= 0 {
			continue
		}
		dateStr := key[:slashIdx]
		model := key[slashIdx+1:]
		result[dateStr][model] = DailyModelStats{
			InputTokens:  ds.input,
			OutputTokens: ds.output,
			TotalTokens:  ds.input + ds.output,
			Requests:     ds.requests,
		}
	}

	return result
}

// stringsIndex is a minimal dependency-free helper to find the first index of
// a substring, avoiding an extra import for "strings" when it's not needed.
func stringsIndex(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// MarshalJSON implements json.Marshaler for dailyModelStat.
func (d *dailyModelStat) MarshalJSON() ([]byte, error) {
	return json.Marshal(dailyModelStatJSON{
		Input:    d.input,
		Output:   d.output,
		Requests: d.requests,
	})
}

// UnmarshalJSON implements json.Unmarshaler for dailyModelStat.
func (d *dailyModelStat) UnmarshalJSON(data []byte) error {
	var j dailyModelStatJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	d.input = j.Input
	d.output = j.Output
	d.requests = j.Requests
	return nil
}
