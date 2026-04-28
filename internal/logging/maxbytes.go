package logging

import "net/http"

// MaxBytes wraps the request body with http.MaxBytesReader so any handler
// reading r.Body cannot consume more than limit bytes. When limit <= 0 the
// middleware is a no-op. Methods without a body (GET/HEAD/DELETE) and CONNECT
// are passed through unchanged.
func MaxBytes(limit int64, next http.Handler) http.Handler {
	if limit <= 0 {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil && r.ContentLength != 0 {
			r.Body = http.MaxBytesReader(w, r.Body, limit)
		}
		next.ServeHTTP(w, r)
	})
}
