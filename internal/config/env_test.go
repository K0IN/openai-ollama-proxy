package config

import (
	"os"
	"testing"
)

func TestExpandEnvVars(t *testing.T) {
	// Set up test environment variables
	_ = os.Setenv("TEST_VAR", "test_value")
	_ = os.Setenv("TEST_EMPTY", "")
	defer func() {
		_ = os.Unsetenv("TEST_VAR")
		_ = os.Unsetenv("TEST_EMPTY")
		_ = os.Unsetenv("TEST_MISSING")
	}()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "no_placeholder",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "simple_placeholder",
			input:    "${TEST_VAR}",
			expected: "test_value",
		},
		{
			name:     "missing_variable",
			input:    "${TEST_MISSING}",
			expected: "",
		},
		{
			name:     "default_value_used",
			input:    "${TEST_MISSING:-default}",
			expected: "default",
		},
		{
			name:     "default_value_ignored_when_set",
			input:    "${TEST_VAR:-default}",
			expected: "test_value",
		},
		{
			name:     "empty_variable_with_default",
			input:    "${TEST_EMPTY:-default}",
			expected: "default",
		},
		{
			name:     "multiple_placeholders",
			input:    "${TEST_VAR} and ${TEST_MISSING:-fallback}",
			expected: "test_value and fallback",
		},
		{
			name:     "partial_line",
			input:    "api_key = ${TEST_VAR}",
			expected: "api_key = test_value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := expandEnvVars(tc.input)
			if result != tc.expected {
				t.Errorf("expandEnvVars(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}
