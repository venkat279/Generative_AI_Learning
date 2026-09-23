# Runbook: Disk Space Alert

## Symptoms

- A "disk usage above 90%" alert fires for a service host.
- Log write failures (`no space left on device`) may follow if usage
  reaches 100%.
- Log rotation or compaction jobs may start failing silently before
  any explicit alert fires.

## Likely Root Cause

Usually an unrotated or unbounded log file, a runaway temp-file
writer, or a build-up of old backups/snapshots on the same volume.

## Diagnostic Steps

1. Run `du -sh` (or the platform equivalent) on the affected volume's
   top-level directories to find what's consuming space.
2. Check whether log rotation is configured and actually running for
   the largest offenders.
3. Check for orphaned temp files left behind by a crashed process.

## Fix

- Immediate mitigation: delete or archive the largest safe-to-remove
  files to free headroom.
- Root fix: configure or fix log rotation, and add a scheduled cleanup
  job for temp files and old backups.
- Preventive: alert at a lower threshold (e.g. 75%) to give more lead
  time before it becomes urgent.
