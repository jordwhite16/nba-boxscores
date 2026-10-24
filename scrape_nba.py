#!/usr/bin/env python3
"""Nightly NBA box-score scraper.

Runs on a GitHub Actions runner, reads a public JSON feed, and commits a parquet
file in the exact schema the projection pipeline expects. No API key, no HTML
scraping, no browser.

WHY ESPN AND NOT THE NBA'S OWN FEED
-----------------------------------
The first version read cdn.nba.com. It works from a home connection and returns
403 Forbidden from a GitHub runner: that CDN blocks datacentre IP ranges, and
Actions runners live in Azure. No user-agent or referer trick gets round an
address-based block, so the feed had to change rather than the request. ESPN's
site API serves the same box scores and does not block datacentre traffic.

The NBA endpoints are kept below as fallbacks, and --probe reports which hosts
actually answer from wherever this is running. If ESPN ever closes too, one red
run will say so in plain text instead of leaving us guessing.

DESIGN RULES, LEARNED THE HARD WAY
----------------------------------
  * Every game is checked against its own published final score before being
    written. A box score that does not foot to the scoreboard is a parsing
    fault, and a wrong number that looks ordinary is worse than a missing one.
  * The script FAILS LOUDLY. A game that is final but cannot be fetched or does
    not validate exits non-zero, so the run goes red and sends an email rather
    than silently committing yesterday's file again.
  * It is idempotent. Re-running adds nothing and breaks nothing, so a retry or
    a manual run is always safe.

USAGE
-----
  python scrape_nba.py                  yesterday and today (the nightly job)
  python scrape_nba.py --since 2026-10-21   every date from then until today
  python scrape_nba.py --probe          report which feeds are reachable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

OUT = os.path.join('data', 'playerlog_2027.parquet')
LOG = os.path.join('data', 'ingest_log.csv')

# With no stored data at all, the first run collects the season to date rather
# than only last night. That makes the very first run the backfill, so the
# season does not have to be assembled by hand to get started.
SEASON_START = '2026-10-20'

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/125.0 Safari/537.36')

ESPN_SB = ('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/'
           'scoreboard?dates={yyyymmdd}&limit=100')
ESPN_BOX = ('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/'
            'summary?event={eid}')

# Probed, not used, unless ESPN stops answering. Kept so a failure is diagnosed
# by the script rather than by me guessing from a screenshot.
PROBES = [
    ('ESPN scoreboard', ESPN_SB.format(yyyymmdd='20260101')),
    ('NBA cdn schedule',
     'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json'),
    ('NBA cdn scoreboard',
     'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'),
    ('data.nba.net', 'https://data.nba.net/prod/v1/today.json'),
]

# ESPN's abbreviations differ from the NBA tricodes the rest of the pipeline
# uses. Anything not listed here is already identical.
TRI = {'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'NO': 'NOP',
       'UTAH': 'UTA', 'WSH': 'WAS'}

# The pipeline reads `position` as a position GROUP and treats a non-empty value
# as "this player started". ESPN gives a position to everyone, so it is only
# written for actual starters, matching the convention the stored season uses.
POSGROUP = {'PG': 'G', 'SG': 'G', 'G': 'G',
            'SF': 'F', 'PF': 'F', 'F': 'F',
            'C': 'C'}

# ESPN labels each stat column; parse by key name rather than by position, so a
# reordered or added column cannot silently shift every number one to the left.
PAIRS = {
    'fieldGoalsMade-fieldGoalsAttempted': ('fieldGoalsMade', 'fieldGoalsAttempted'),
    'threePointFieldGoalsMade-threePointFieldGoalsAttempted':
        ('threePointersMade', 'threePointersAttempted'),
    'freeThrowsMade-freeThrowsAttempted': ('freeThrowsMade', 'freeThrowsAttempted'),
}
SINGLES = {
    'points': 'points', 'rebounds': 'reboundsTotal',
    'offensiveRebounds': 'reboundsOffensive', 'assists': 'assists',
    'steals': 'steals', 'blocks': 'blocks', 'turnovers': 'turnovers',
    'fouls': 'foulsPersonal',
}
# Fallback if ESPN ever drops `keys` and leaves only the display labels.
LABELS = {'PTS': 'points', 'REB': 'reboundsTotal', 'OREB': 'reboundsOffensive',
          'AST': 'assists', 'STL': 'steals', 'BLK': 'blocks', 'TO': 'turnovers',
          'PF': 'foulsPersonal', 'MIN': 'minutes',
          'FG': 'fieldGoalsMade-fieldGoalsAttempted',
          '3PT': 'threePointFieldGoalsMade-threePointFieldGoalsAttempted',
          'FT': 'freeThrowsMade-freeThrowsAttempted'}


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def get(url: str, tries: int = 4) -> dict:
    """Fetch JSON with a browser user agent and a polite backoff."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': UA,
                'Accept': 'application/json',
            })
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:                      # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f'could not fetch {url}: {last}')


def probe() -> int:
    """Report which feeds answer from here. Printed automatically on failure."""
    print('\nfeed reachability from this runner:')
    for name, url in PROBES:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f'  {name:22} HTTP {r.status}  ok')
        except Exception as e:                      # noqa: BLE001
            print(f'  {name:22} {type(e).__name__}: {e}')
    return 0


def eastern_date(offset_days: int = 0) -> str:
    """NBA games are dated by their US Eastern tip-off, not by UTC. A 10:30pm
    Pacific game is still 'yesterday' well into the following UTC day, so using
    UTC here would silently skip the late west-coast games."""
    et = datetime.now(timezone.utc) - timedelta(hours=5)
    return (et + timedelta(days=offset_days)).strftime('%Y-%m-%d')


def tri(abbrev: str) -> str:
    a = (abbrev or '').upper()
    return TRI.get(a, a)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def games_on(date: str) -> list[dict]:
    """Every game on one US Eastern date, with its published final score."""
    sb = get(ESPN_SB.format(yyyymmdd=date.replace('-', '')))
    out = []
    for ev in sb.get('events', []):
        comp = (ev.get('competitions') or [{}])[0]
        st = ((ev.get('status') or {}).get('type') or {})
        g = dict(gameId=str(ev.get('id')), date=date,
                 final=bool(st.get('completed')),
                 state=st.get('state'), away=None, home=None,
                 away_pts=None, home_pts=None)
        for c in comp.get('competitors', []):
            side = 'home' if c.get('homeAway') == 'home' else 'away'
            g[side] = tri((c.get('team') or {}).get('abbreviation'))
            try:
                g[f'{side}_pts'] = int(c.get('score'))
            except (TypeError, ValueError):
                g[f'{side}_pts'] = None
        if g['away'] and g['home']:
            out.append(g)
    return out


def _stat_map(block: dict) -> list[str]:
    """The column names for this team's athlete rows, normalised to ESPN keys."""
    keys = block.get('keys')
    if keys:
        return list(keys)
    return [LABELS.get(str(n).upper(), '') for n in (block.get('names') or [])]


def _num(v) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return 0


def _minutes(v) -> float:
    """ESPN prints whole minutes ('34'); tolerate 'MM:SS' in case that changes."""
    s = str(v or '').strip()
    if ':' in s:
        a, b = (s.split(':') + ['0'])[:2]
        try:
            return float(a) + float(b) / 60
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_box(g: dict) -> pd.DataFrame | None:
    """One game's player rows, or None if the feed has not published it."""
    try:
        d = get(ESPN_BOX.format(eid=g['gameId']))
    except RuntimeError:
        return None
    teams = ((d.get('boxscore') or {}).get('players') or [])
    if not teams:
        return None

    rows = []
    for t in teams:
        abbr = tri((t.get('team') or {}).get('abbreviation'))
        for block in (t.get('statistics') or []):
            cols = _stat_map(block)
            for a in (block.get('athletes') or []):
                if a.get('didNotPlay'):
                    continue
                stats = a.get('stats') or []
                if len(stats) != len(cols):
                    continue                 # shape mismatch: skip, validation catches it
                vals = dict(zip(cols, stats))
                mins = _minutes(vals.get('minutes'))
                if mins <= 0:
                    continue                 # did not play: no information
                ath = a.get('athlete') or {}
                row = dict(
                    gameId=g['gameId'], date=g['date'],
                    away=g['away'], home=g['home'],
                    teamTricode=abbr,
                    # ESPN ids live in a different namespace from NBA personIds
                    # and the ranges overlap, so they are stored NEGATIVE. That
                    # makes a false match with a stored NBA id impossible, and
                    # the blend's name fallback does the joining instead.
                    personId=-_num(ath.get('id')),
                    firstName=ath.get('firstName') or '',
                    familyName=ath.get('lastName') or '',
                    displayName=ath.get('displayName') or '',
                    mins=mins,
                    starter=bool(a.get('starter')),
                    position=(POSGROUP.get(
                        str(((ath.get('position') or {}).get('abbreviation')
                             or '')).upper(), '') if a.get('starter') else ''),
                )
                for k, (made, att) in PAIRS.items():
                    raw = str(vals.get(k, '')).split('-')
                    row[made] = _num(raw[0]) if raw else 0
                    row[att] = _num(raw[1]) if len(raw) > 1 else 0
                for k, name in SINGLES.items():
                    row[name] = _num(vals.get(k))
                rows.append(row)
    return pd.DataFrame(rows) if rows else None


def derive(d: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the projection pipeline needs that no box score prints:
    usage rate, possessions and pace. Possessions are stored PER PLAYER, the way
    the reference dataset does it, so a team-game sums to roughly five times the
    team total; storing the team figure on every row puts the season on a
    different scale and the error only shows once two seasons are blended."""
    tm = d.groupby(['gameId', 'teamTricode']).apply(
        lambda g: pd.Series({
            'tposs': g.fieldGoalsAttempted.sum() + 0.44 * g.freeThrowsAttempted.sum()
                     - g.reboundsOffensive.sum() + g.turnovers.sum(),
            'tmin': g.mins.sum(),
            'tfga': g.fieldGoalsAttempted.sum(),
            'tfta': g.freeThrowsAttempted.sum(),
            'ttov': g.turnovers.sum()}), include_groups=False)
    d = d.merge(tm, left_on=['gameId', 'teamTricode'], right_index=True, how='left')

    usg = (100 * ((d.fieldGoalsAttempted + 0.44 * d.freeThrowsAttempted + d.turnovers)
                  * (d.tmin / 5))
           / (d.mins * (d.tfga + 0.44 * d.tfta + d.ttov)))
    d['usagePercentage'] = usg.clip(0, 100).fillna(0) / 100.0
    d['possessions'] = d.tposs * d.mins / 48.0
    d['pace'] = 48 * d.tposs / (d.tmin / 5)
    d['minutes'] = d.mins.map(lambda m: f'{int(m)}:{int(round((m % 1) * 60)):02d}')
    return d.drop(columns=['tposs', 'tmin', 'tfga', 'tfta', 'ttov', 'mins'])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def already_have() -> tuple[set, set]:
    """What is stored, by game id AND by fixture. The fixture key matters because
    the 16 games ingested by hand carry NBA game ids, and ESPN's ids for the same
    games are different numbers: without it a backfill would duplicate them."""
    if not os.path.exists(OUT):
        return set(), set()
    df = pd.read_parquet(OUT)
    ids = set(df.gameId.astype(str).unique())
    fx = set()
    if {'date', 'away', 'home'} <= set(df.columns):
        fx = set((df.date.astype(str) + '|' + df.away.astype(str)
                  + '@' + df.home.astype(str)).unique())
    return ids, fx


def run(dates: list[str]) -> int:
    print(f'checking US Eastern dates: {dates}')
    sched = []
    for d in dates:
        try:
            sched += games_on(d)
        except RuntimeError as e:
            print(f'FATAL: {e}', file=sys.stderr)
            probe()
            return 1
        time.sleep(0.3)

    finals = [g for g in sched if g['final']]
    print(f'{len(sched)} games scheduled, {len(finals)} final')
    if not finals:
        print('nothing final yet; nothing to do')
        return 0

    ids, fx = already_have()
    todo = [g for g in finals
            if g['gameId'] not in ids
            and f"{g['date']}|{g['away']}@{g['home']}" not in fx]
    print(f'{len(ids)} games already stored, {len(todo)} to add')
    if not todo:
        return 0

    frames, failed, log = [], [], []
    for g in todo:
        box = parse_box(g)
        time.sleep(0.3)
        label = f"{g['date']} {g['away']}@{g['home']}"
        if box is None or box.empty:
            failed.append((g['gameId'], label, 'no box score published'))
            continue

        ok = True
        for side in ('away', 'home'):
            want = g[f'{side}_pts']
            got = int(box.loc[box.teamTricode == g[side], 'points'].sum())
            if want is None or got != int(want):
                failed.append((g['gameId'], label,
                               f'{g[side]} sums to {got}, scoreboard says {want}'))
                ok = False
        if not ok:
            continue

        # soft check: a team-game should be 240 player-minutes. Not fatal, because
        # ESPN rounds to whole minutes, but worth seeing in the log if it drifts.
        for side in ('away', 'home'):
            tm = box.loc[box.teamTricode == g[side], 'mins'].sum()
            if abs(tm - 240) > 6:
                print(f'  note {label}: {g[side]} has {tm:.0f} player-minutes, not ~240')

        frames.append(box)
        log.append(dict(gameId=g['gameId'], date=g['date'],
                        fixture=f"{g['away']}@{g['home']}",
                        score=f"{g['away_pts']}-{g['home_pts']}", players=len(box),
                        ingested=datetime.now(timezone.utc).isoformat(timespec='seconds')))
        print(f"  ok {label} {g['away_pts']}-{g['home_pts']} ({len(box)} players)")

    if frames:
        new = derive(pd.concat(frames, ignore_index=True))
        os.makedirs('data', exist_ok=True)
        if os.path.exists(OUT):
            new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
            new = new.drop_duplicates(subset=['gameId', 'personId'], keep='last')
        new.to_parquet(OUT, index=False)
        pd.DataFrame(log).to_csv(
            LOG, mode='a', header=not os.path.exists(LOG), index=False)
        print(f'\nwrote {OUT}: {new.gameId.nunique()} games, {len(new)} player-games')

    if failed:
        print('\nFAILED:', file=sys.stderr)
        for gid, fx_, why in failed:
            print(f'  {gid} {fx_}: {why}', file=sys.stderr)
        # a game that is final but did not validate is a real fault. Going red
        # here is the point: a silent miss leaves the model running on stale
        # baselines with nobody any the wiser.
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', help='backfill from this US Eastern date, YYYY-MM-DD')
    ap.add_argument('--probe', action='store_true', help='report feed reachability')
    a = ap.parse_args()

    if a.probe:
        return probe()

    since = a.since
    if not since and not os.path.exists(OUT):
        since = SEASON_START
        print(f'no stored data yet: backfilling the season from {since}')

    if since:
        start = datetime.strptime(since, '%Y-%m-%d').date()
        end = datetime.strptime(eastern_date(0), '%Y-%m-%d').date()
        if start > end:
            print('--since is in the future', file=sys.stderr)
            return 1
        dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range((end - start).days + 1)]
    else:
        # yesterday and today in US Eastern: late west-coast games finish after
        # midnight Eastern, so a single day would miss them
        dates = sorted({eastern_date(-1), eastern_date(0)})

    return run(dates)


if __name__ == '__main__':
    sys.exit(main())
