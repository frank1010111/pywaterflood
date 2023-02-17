"""Nox sessions for linting, docs, and testing."""
from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter.

    Includes all the pre-commit checks on all the files.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("maturin")
    # You have to run `maturin develop` to avoid this: https://github.com/PyO3/maturin/issues/490
    session.run("maturin", "develop", "--release", "--extras=test")
    session.run(
        "pytest",
        "--cov=pywaterflood",
        "--cov-append",
        "--cov-report=xml",
        *session.posargs,
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install(".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")  # noqa: T201
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            session.warn("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and a wheel."""
    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build", "-o", "build")
