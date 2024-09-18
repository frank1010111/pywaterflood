"""Nox sessions for linting, docs, and testing."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"


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
    session.install(".[test]")
    session.run(
        "pytest",
        "--cov=pywaterflood",
        "--cov-append",
        "--cov-report=xml",
        *session.posargs,
    )


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-e.[docs]", *extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run("sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs)
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and a wheel."""
    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build", "-o", "build")


@nox.session
def paper(session: nox.Sesson) -> None:
    """Build the JOSS paper draft."""
    paper_dir = DIR.joinpath("paper")
    session.run(
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{paper_dir}:/data",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "JOURNAL=joss",
        "openjournals/inara",
        external=True,
    )
