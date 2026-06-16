package config

import (
	"strings"
	"testing"
)

func TestModelMappingValidate(t *testing.T) {
	tests := []struct {
		name    string
		m       ModelMapping
		wantErr string
	}{
		{
			name:    "valid",
			m:       ModelMapping{Upstream: "gpt-4o", Local: "gpt-4o", ContextLength: 128000},
			wantErr: "",
		},
		{
			name:    "valid with zero context_length",
			m:       ModelMapping{Upstream: "gpt-4o", Local: "gpt-4o"},
			wantErr: "",
		},
		{
			name:    "empty upstream",
			m:       ModelMapping{Upstream: "", Local: "local"},
			wantErr: "empty upstream name",
		},
		{
			name:    "empty local",
			m:       ModelMapping{Upstream: "upstream", Local: ""},
			wantErr: "empty local name",
		},
		{
			name:    "negative context_length",
			m:       ModelMapping{Upstream: "gpt-4o", Local: "gpt-4o", ContextLength: -1},
			wantErr: "context_length must be >= 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.m.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tt.wantErr)
				}
			}
		})
	}
}

func TestUpstreamConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		u       UpstreamConfig
		wantErr string
	}{
		{
			name: "valid",
			u: UpstreamConfig{
				URL: "http://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "",
		},
		{
			name: "empty URL",
			u: UpstreamConfig{
				URL: "",
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "URL must not be empty",
		},
		{
			name: "invalid scheme",
			u: UpstreamConfig{
				URL: "ftp://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "must use http or https",
		},
		{
			name: "missing host",
			u: UpstreamConfig{
				URL: "http:///path",
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "missing a host",
		},
		{
			name: "no models",
			u: UpstreamConfig{
				URL:    "http://localhost:8000",
				Models: nil,
			},
			wantErr: "must define at least one model",
		},
		{
			name: "empty models slice",
			u: UpstreamConfig{
				URL:    "http://localhost:8000",
				Models: []ModelMapping{},
			},
			wantErr: "must define at least one model",
		},
		{
			name: "invalid model mapping",
			u: UpstreamConfig{
				URL: "http://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "", Local: "local"},
				},
			},
			wantErr: "empty upstream name",
		},
		{
			name: "passthrough valid — no api_key",
			u: UpstreamConfig{
				URL:         "http://localhost:8000",
				Passthrough: true,
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "",
		},
		{
			name: "passthrough with api_key — conflict",
			u: UpstreamConfig{
				URL:         "http://localhost:8000",
				APIKey:      "sk-abc",
				Passthrough: true,
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o"},
				},
			},
			wantErr: "cannot both be set",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.u.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tt.wantErr)
				}
			}
		})
	}
}

func TestBuildRoutingTable(t *testing.T) {
	t.Run("valid with multiple upstreams", func(t *testing.T) {
		upstreams := []UpstreamConfig{
			{
				URL: "http://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "qwen2.5-coder-14b", Local: "qwen-coder", ContextLength: 32768},
					{Upstream: "qwen3-27b-fp8", Local: "qwen3-large"},
				},
			},
			{
				URL:    "https://api.openai.com",
				APIKey: "sk-abc",
				Models: []ModelMapping{
					{Upstream: "gpt-4o", Local: "gpt-4o", ContextLength: 128000},
				},
			},
		}

		rt, err := BuildRoutingTable(upstreams, 65536)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Check qwen-coder with explicit context_length override
		entry, ok := rt.Lookup("qwen-coder")
		if !ok {
			t.Fatal("qwen-coder not found")
		}
		if entry.URL != "http://localhost:8000" {
			t.Fatalf("qwen-coder URL = %q", entry.URL)
		}
		if entry.UpstreamModel != "qwen2.5-coder-14b" {
			t.Fatalf("qwen-coder upstream model = %q", entry.UpstreamModel)
		}
		if entry.ContextLength != 32768 {
			t.Fatalf("qwen-coder context_length = %d, want 32768", entry.ContextLength)
		}

		// Check qwen3-large with global default
		entry, ok = rt.Lookup("qwen3-large")
		if !ok {
			t.Fatal("qwen3-large not found")
		}
		if entry.ContextLength != 65536 {
			t.Fatalf("qwen3-large context_length = %d, want 65536 (global default)", entry.ContextLength)
		}

		// Check gpt-4o with API key
		entry, ok = rt.Lookup("gpt-4o")
		if !ok {
			t.Fatal("gpt-4o not found")
		}
		if entry.APIKey != "sk-abc" {
			t.Fatalf("gpt-4o API key = %q", entry.APIKey)
		}
		if entry.ContextLength != 128000 {
			t.Fatalf("gpt-4o context_length = %d, want 128000", entry.ContextLength)
		}

		// Check AllModels sorted
		models := rt.AllModels()
		if len(models) != 3 {
			t.Fatalf("AllModels length = %d, want 3", len(models))
		}
		if models[0] != "gpt-4o" || models[1] != "qwen-coder" || models[2] != "qwen3-large" {
			t.Fatalf("AllModels = %v, want sorted", models)
		}

		// Check AllUpstreams
		if len(rt.AllUpstreams()) != 2 {
			t.Fatalf("AllUpstreams length = %d, want 2", len(rt.AllUpstreams()))
		}

		// Check missing model
		_, ok = rt.Lookup("nonexistent")
		if ok {
			t.Fatal("nonexistent should not be found")
		}
	})

	t.Run("passthrough propagation", func(t *testing.T) {
		upstreams := []UpstreamConfig{
			{
				URL:         "http://localhost:8000",
				Passthrough: true,
				Models: []ModelMapping{
					{Upstream: "m1", Local: "m1"},
				},
			},
		}

		rt, err := BuildRoutingTable(upstreams, 65536)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		entry, ok := rt.Lookup("m1")
		if !ok {
			t.Fatal("m1 not found")
		}
		if !entry.Passthrough {
			t.Fatal("Passthrough should be true")
		}
		if entry.APIKey != "" {
			t.Fatalf("APIKey should be empty, got %q", entry.APIKey)
		}

		// Verify AllUpstreams also carries the flag.
		all := rt.AllUpstreams()
		if len(all) != 1 || !all[0].Passthrough {
			t.Fatal("AllUpstreams passthrough flag not propagated")
		}
	})

	t.Run("duplicate local model names", func(t *testing.T) {
		upstreams := []UpstreamConfig{
			{
				URL: "http://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "model-a", Local: "shared-name"},
				},
			},
			{
				URL: "http://other:9000",
				Models: []ModelMapping{
					{Upstream: "model-b", Local: "shared-name"},
				},
			},
		}

		_, err := BuildRoutingTable(upstreams, 65536)
		if err == nil {
			t.Fatal("expected error for duplicate local model name")
		}
		if !strings.Contains(err.Error(), "duplicate local model") {
			t.Fatalf("error %q does not contain 'duplicate local model'", err.Error())
		}
	})

	t.Run("upstream validation failure", func(t *testing.T) {
		upstreams := []UpstreamConfig{
			{
				URL:    "not-a-valid-url://",
				Models: []ModelMapping{{Upstream: "m", Local: "m"}},
			},
		}

		_, err := BuildRoutingTable(upstreams, 65536)
		if err == nil {
			t.Fatal("expected error for invalid upstream")
		}
	})

	t.Run("zero context_length means use global", func(t *testing.T) {
		upstreams := []UpstreamConfig{
			{
				URL: "http://localhost:8000",
				Models: []ModelMapping{
					{Upstream: "m", Local: "m", ContextLength: 0},
				},
			},
		}

		rt, err := BuildRoutingTable(upstreams, 99999)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		entry, _ := rt.Lookup("m")
		if entry.ContextLength != 99999 {
			t.Fatalf("context_length = %d, want 99999 (global default)", entry.ContextLength)
		}
	})
}
