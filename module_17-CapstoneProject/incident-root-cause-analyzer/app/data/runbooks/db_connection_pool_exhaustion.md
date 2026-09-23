# Runbook: Database Connection Pool Exhaustion

## Symptoms

- `payments-service` (or any service backed by `postgres-db`) logs
  `could not acquire connection from pool` or `connection pool
  exhausted` errors.
- `postgres-db` logs `FATAL: sorry, too many clients already` or
  `remaining connection slots are reserved for non-replication
  superuser connections`.
- Downstream `api-gateway` starts returning `502 Bad Gateway` for
  requests that depend on the affected service.
- Error rate on the affected service spikes sharply (often above 50%)
  while CPU usage stays only moderately elevated — this is a
  connection-starvation problem, not a compute problem.

## Likely Root Cause

A recent deploy introduced a code path that opens a database
connection without releasing it back to the pool (a connection leak),
or a spike in traffic exceeded the configured pool size. Once the pool
is fully checked out, every new request queues and eventually times
out, and the leaked connections are never returned, so the pool never
recovers on its own.

## Diagnostic Steps

1. Check the affected service's connection pool stats log line
   (checked out / idle / waiting) to confirm exhaustion.
2. Check whether a deploy happened shortly before the first error —
   correlate deploy timestamps with the first pool-exhaustion error.
3. Check `postgres-db`'s connection slot usage to confirm the database
   side is also seeing the same exhaustion.

## Fix

- Immediate mitigation: restart the affected service's instances to
  force-release leaked connections.
- Root fix: roll back the deploy that introduced the leak, or add an
  explicit connection-release (e.g. a `finally`/context-manager block)
  around the code path that was leaking.
- Preventive: add pool-utilization alerting well below 100% so this is
  caught before the pool is fully exhausted.
