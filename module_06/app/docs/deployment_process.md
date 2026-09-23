# Deployment Process

All services deploy through the shared CI/CD pipeline. A deployment
begins when a pull request merges into `main`; the pipeline then runs the
full test suite, builds a container image, and pushes it to the internal
registry.

Deployments to staging happen automatically on every merge. Deployments
to production require a manual approval step from a second engineer and
are limited to weekdays between 9am and 4pm, so someone is available to
respond if something goes wrong.

Every production deployment must include a rollback plan in the PR
description. Rolling back is done by re-deploying the previous container
image tag through the same pipeline - it takes about three minutes and
does not require a new build.

Feature flags should be used for any change that alters user-facing
behavior, so a bad deploy can be mitigated by disabling the flag instead
of rolling back the whole release.
