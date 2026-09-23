# Engineering Onboarding Checklist

Before your first day, IT should have provisioned your laptop and sent an
invite to the internal Slack workspace. If either is missing, contact
the helpdesk before you start.

On day one: get access to the main GitHub organization, request VPN
credentials, and clone the `platform` repository. Run the `./setup.sh`
script in that repo - it installs the required language runtimes, sets
up pre-commit hooks, and configures your local environment file from the
shared template.

By the end of week one, you should have shipped a small change through
the full deployment process (see the deployment process document) with a
buddy reviewing your PR, so you experience the pipeline end to end while
someone experienced is watching.

By the end of week two, you should be added to the on-call rotation as a
shadow (see the on-call rotation document) for one rotation before taking
a primary shift yourself.
