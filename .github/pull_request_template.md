# Description

This is a [choose one]

- feature addition
- documentation improvement
- bugfix

It enhances/documents/fixes [something]

## Pull request checklist

- [ ] If features have changed, there's new documentation, and it has been checked: `nox -s docs -- --serve`
- [ ] If a bugfix, new tests are in `testing/`
- Passes all CI/CD tests:
  - [ ] Pre-commit linting passes: `nox -s lint`
  - [ ] Package builds `nox -s build`
  - [ ] Tests all pass: `nox -s tests`
