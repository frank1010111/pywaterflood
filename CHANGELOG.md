# Changelog

<!--next-version-placeholder-->

## v0.3.2 (11/10/2023)

**Critical fix:**

- :bug: fix import error pitfall for maturin by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/62

**Python version update:**

- :test_tube: bump latest tested python to 3.12 by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/60

**Added tests:**

- Add tests by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/56
- Add tests by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/57

**Dependency updates:**

- build(deps): bump pypa/cibuildwheel from 2.15.0 to 2.16.0 by @dependabot in https://github.com/frank1010111/pywaterflood/pull/55
- build(deps): bump pypa/cibuildwheel from 2.16.0 to 2.16.1 by @dependabot in https://github.com/frank1010111/pywaterflood/pull/59
- :memo: :bug: :hammer: change from myst_nb to myst_parser for docs to … by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/63

**Full Changelog**: https://github.com/frank1010111/pywaterflood/compare/v0.3.1...v0.3.2

## v0.3.1 (16/09/2023)

**Code improvements**:

- Fix MPI to be symmetric (https://github.com/frank1010111/pywaterflood/pull/47)
- Cleaner rust import (https://github.com/frank1010111/pywaterflood/pull/54)

**Documentation**:

- 35 update docs by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/36
- :memo: add MPI example by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/47
- Submit to joss by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/50

## v0.3.0 (03/05/2023)

Code improvements:

- New Rust computational backend in https://github.com/frank1010111/pywaterflood/pull/25

Non-code improvements:

- Pre commit changes by @frank1010111 in https://github.com/frank1010111/pywaterflood/pull/27
- update pandas version to fix numpy incompatiblility

## v0.2.0 (21/05/2022)

### Features

- Added `q_BHP` and `CRMCompensated` to perform copensated CRM where bottom hole
  pressure variations are taken into account.
- Updated and simplified dockerfile

### Bugfixes

- Fixed testing bugs

### Deprecated

- support for python 3.6

## v0.1.1 (3/11/2021)

- The great renaming to `pywaterflood`

### Features

- multiwell productivity index added

## v0.1.0 (23/08/2021)

- First pip build of `pyCRM`!

## v0.0.1 (15/01/2020)

- First release of `pyCRM`!
