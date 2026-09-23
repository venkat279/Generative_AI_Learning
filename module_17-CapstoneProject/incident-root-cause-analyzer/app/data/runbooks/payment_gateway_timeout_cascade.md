# Runbook: Downstream Payment Gateway Timeout Cascade

## Symptoms

- `payments-service` logs repeated `external payment gateway request
  timed out` errors, followed by `thread pool exhausted waiting on
  external payment gateway`.
- `api-gateway` starts returning `504 Gateway Timeout` and eventually
  `500 Internal Server Error` (`thread pool exhausted`) for
  `/checkout` shortly after.
- `checkout-service` logs `payment step failed` errors referencing
  upstream `504`/`500`.
- Error rate and p99 latency spike together on both `payments-service`
  and `api-gateway`, while `postgres-db` and `redis-cache` stay
  healthy — the failure originates outside the stack entirely, at a
  third-party dependency.

## Likely Root Cause

A third-party/external payment gateway that `payments-service` calls
synchronously started responding slowly or not at all. Because the
call has no circuit breaker or aggressive timeout, every request
thread that calls the gateway blocks until it times out, and enough
concurrent requests do this simultaneously to exhaust the service's
thread pool — which then cascades upstream to `api-gateway` and
`checkout-service` as timeouts and 5xx errors, even though neither of
those services has a problem of its own.

## Diagnostic Steps

1. Check `payments-service` logs for repeated external-gateway timeout
   errors as the first symptom in the timeline.
2. Confirm `postgres-db` and `redis-cache` show no corresponding
   errors in the same window — this rules out an internal-dependency
   cause.
3. Check whether the timeout errors stop and latency recovers on their
   own after a period, consistent with a transient third-party outage
   rather than a code regression.

## Fix

- Immediate mitigation: none available in-band since the failure is
  external — wait for the gateway to recover, or fail over to a
  backup payment provider if one is configured.
- Root fix: add a circuit breaker around the external gateway call so
  a slow/down dependency fails fast instead of exhausting the thread
  pool, and reduce the per-call timeout well below the pool's
  saturation point.
- Preventive: queue payment attempts for async retry instead of
  holding a request thread open for the full external call.
