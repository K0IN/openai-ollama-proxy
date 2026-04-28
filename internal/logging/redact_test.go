package logging

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRedactHeaderValue_Sensitive(t *testing.T) {
	cases := []string{"Authorization", "authorization", "AUTHORIZATION", "Cookie", "X-Api-Key", "Api-Key", "X-Auth-Token", "Proxy-Authorization", "Set-Cookie"}
	for _, name := range cases {
		got := RedactHeaderValue(name, "Bearer sk-secret-123")
		if got != "[REDACTED]" {
			t.Errorf("RedactHeaderValue(%q) = %q, want [REDACTED]", name, got)
		}
	}
}

func TestRedactHeaderValue_NotSensitive(t *testing.T) {
	got := RedactHeaderValue("Content-Type", "application/json")
	if got != "application/json" {
		t.Errorf("RedactHeaderValue(Content-Type) = %q, want application/json", got)
	}
}

func TestRedactHeaderValue_EmptySensitiveValue(t *testing.T) {
	got := RedactHeaderValue("Authorization", "")
	if got != "" {
		t.Errorf("RedactHeaderValue(Authorization,empty) = %q, want empty", got)
	}
}

func TestRedactJSONForLog_TopLevelKeys(t *testing.T) {
	in := []byte(`{"api_key":"sk-abc","apikey":"xyz","model":"qwen","authorization":"Bearer t","password":"p","secret":"s","token":"tok"}`)
	out := RedactJSONForLog(in)
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("redacted output not valid JSON: %v", err)
	}
	for _, key := range []string{"api_key", "apikey", "authorization", "password", "secret", "token"} {
		if got[key] != "[REDACTED]" {
			t.Errorf("got[%q] = %v, want [REDACTED]", key, got[key])
		}
	}
	if got["model"] != "qwen" {
		t.Errorf("model field should be untouched, got %v", got["model"])
	}
}

func TestRedactJSONForLog_NestedAndArrays(t *testing.T) {
	in := []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"system","api_key":"leaked"}],"meta":{"X-Api-Key":"shh"}}`)
	out := RedactJSONForLog(in)
	if bytes.Contains(out, []byte("leaked")) {
		t.Errorf("nested api_key not redacted: %s", out)
	}
	if bytes.Contains(out, []byte("shh")) {
		t.Errorf("nested X-Api-Key not redacted: %s", out)
	}
	if !bytes.Contains(out, []byte("\"role\":\"user\"")) {
		t.Errorf("non-sensitive fields lost: %s", out)
	}
}

func TestRedactJSONForLog_InvalidJSONReturnedUnchanged(t *testing.T) {
	in := []byte(`not really json`)
	out := RedactJSONForLog(in)
	if string(out) != "not really json" {
		t.Errorf("got %q, want unchanged input", out)
	}
}

func TestRedactJSONForLog_EmptyInput(t *testing.T) {
	if got := RedactJSONForLog(nil); got != nil {
		t.Errorf("nil input should pass through, got %v", got)
	}
	if got := RedactJSONForLog([]byte{}); len(got) != 0 {
		t.Errorf("empty input should pass through, got %v", got)
	}
}

func TestRedactJSONForLog_PreservesNonStringValues(t *testing.T) {
	in := []byte(`{"temperature":0.7,"stream":true,"model":"qwen","messages":null}`)
	out := RedactJSONForLog(in)
	if !strings.Contains(string(out), "0.7") {
		t.Errorf("number value lost: %s", out)
	}
	if !strings.Contains(string(out), "true") {
		t.Errorf("bool value lost: %s", out)
	}
}
