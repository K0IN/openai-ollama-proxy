package logging

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMaxBytes_BelowLimit_PassesThrough(t *testing.T) {
	var got []byte
	handler := MaxBytes(100, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("hello"))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	if string(got) != "hello" {
		t.Fatalf("body = %q, want %q", got, "hello")
	}
}

func TestMaxBytes_OverLimit_TripsReader(t *testing.T) {
	var readErr error
	handler := MaxBytes(4, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, readErr = io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, readErr.Error(), http.StatusRequestEntityTooLarge)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("toolong"))
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if readErr == nil {
		t.Fatalf("expected MaxBytesReader to surface an error reading oversize body")
	}
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", rr.Code)
	}
}

func TestMaxBytes_ZeroLimit_NoOp(t *testing.T) {
	wrapped := MaxBytes(0, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if string(body) != "anything" {
			t.Errorf("body = %q, want %q", body, "anything")
		}
	}))
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader("anything"))
	wrapped.ServeHTTP(httptest.NewRecorder(), req)
}
