# Coding Standards

Every pull request needs at least one approval before merging, and CI
must be green - no merging with failing tests or linter errors, even for
"trivial" changes.

Keep pull requests small and focused on one change. A PR that mixes a
refactor with a behavior change is harder to review and harder to revert
safely if something breaks; split them into separate PRs when possible.

Write tests for new behavior, not just for bug fixes. A change without
test coverage is treated as incomplete during review, and reviewers are
expected to ask for tests rather than approve without them.

Commit messages should explain *why* a change was made, not just what
changed - the diff already shows what changed. Avoid vague messages like
"fix bug" or "update code" with no further context.

Don't introduce a new third-party dependency without a short note in the
PR description explaining why an existing library couldn't do the job.
