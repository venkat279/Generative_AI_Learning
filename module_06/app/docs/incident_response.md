# Incident Response Runbook

If you notice a production outage or major degradation, declare an
incident immediately rather than trying to fix it quietly first. Post in
the `#incidents` channel with a one-line summary, the affected service,
and your best guess at severity (SEV1 = full outage, SEV2 = partial
degradation, SEV3 = minor/cosmetic).

The first responder becomes the incident commander until someone else
explicitly takes over. The incident commander's job is coordination, not
necessarily fixing the bug themselves: pull in the right people, keep the
incident channel updated every 15 minutes, and decide when to escalate.

Mitigate before you diagnose. If rolling back the last deploy or
disabling a feature flag would likely stop the bleeding, do that first
and investigate the root cause afterward.

Every SEV1 or SEV2 incident needs a written postmortem within two
business days. Postmortems are blameless - the goal is fixing the system,
not finding who to blame.
