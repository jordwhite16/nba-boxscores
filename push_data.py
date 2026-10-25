#!/usr/bin/env python3
"""Commit and push whatever the scraper produced, surviving a diverged remote.

This lives in its own file rather than inside the workflow YAML on purpose. The
logic was a thirty-line bash block in the `run:` step, and editing that on a
phone truncated it twice -- once losing a trailing line so `exit 1` ran into a
leftover `fi`. The workflow now calls this script and the YAML step is one line,
so there is almost nothing left to mangle.

WHY IT DOES NOT REBASE
----------------------
The first version ran `git pull --rebase --autostash || true` before pushing and
destroyed a night's collection. data/playerlog_2027.parquet is a BINARY file:
git cannot merge it, so a diverged remote turns the rebase into an unresolvable
conflict. The `|| true` then swallowed that failure and left the repository
mid-rebase on a detached HEAD, and the push died with "You are not currently on
a branch", exit 128 -- taking the already-committed games with the workspace.

Instead, on a rejected push, the REMOTE is treated as the truth: our commit is
discarded, the tree is reset to the remote, and the scraper is re-run on top of
it. The scraper reads the existing file, de-duplicates by game and adds only
what is missing, so re-deriving is exactly equivalent to a merge and cannot
conflict. Verified against a reproduction of the original failure: the old path
lost two games, this path preserved all of them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

DATA = 'data'
SCRAPER = 'scrape_nba.py'
ATTEMPTS = 3


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, echoing it so the Actions log shows what happened."""
    print(f'  $ {" ".join(args)}', flush=True)
    p = subprocess.run(args, capture_output=True, text=True)
    for stream in (p.stdout, p.stderr):
        if stream and stream.strip():
            print('    ' + stream.strip().replace('\n', '\n    '), flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f'::error::{args[0]} {args[1]} failed '
                         f'with exit {p.returncode}')
    return p


def dirty() -> bool:
    """Is there anything uncommitted under data/?"""
    os.makedirs(DATA, exist_ok=True)   # a probe run scrapes nothing and the
    # step runs even after a failure, so the directory may not exist
    p = subprocess.run(['git', 'status', '--porcelain', '--', DATA],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f'::error::git status failed: {p.stderr.strip()}')
    return bool(p.stdout.strip())


def main() -> int:
    branch = os.environ.get('GITHUB_REF_NAME', 'main')
    run(['git', 'config', 'user.name', 'nba-scraper'])
    run(['git', 'config', 'user.email', 'actions@github.com'])

    for attempt in range(1, ATTEMPTS + 1):
        if not dirty():
            print('nothing new to commit')
            return 0

        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        run(['git', 'add', DATA])
        run(['git', 'commit', '-m', f'box scores: {stamp}'])

        push = run(['git', 'push', 'origin', f'HEAD:{branch}'], check=False)
        if push.returncode == 0:
            print(f'pushed on attempt {attempt}')
            return 0

        print(f'::warning::push rejected; re-deriving on top of the remote '
              f'(attempt {attempt} of {ATTEMPTS})')
        run(['git', 'fetch', 'origin', branch])
        run(['git', 'reset', '--hard', f'origin/{branch}'])

        # A failure here is tolerable, unlike the one that caused all this:
        # dirty() at the top of the next pass decides whether there is anything
        # to commit, so a bad scrape leaves the tree clean and the loop exits
        # reporting nothing new rather than pushing a broken state.
        again = run([sys.executable, SCRAPER], check=False)
        if again.returncode != 0:
            print(f'::warning::re-scrape exited {again.returncode} '
                  f'on attempt {attempt}')

    print(f'::error::could not push after {ATTEMPTS} attempts', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
