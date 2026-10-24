#!/usr/bin/env python3
"""Nightly NBA box-score scraper.

Runs on a GitHub Actions runner, reads the NBA's own public JSON feeds, and
commits a parquet file in the exact schema the projection pipeline expects.
No API key, no scraping of HTML, no browser.

Why this exists: the projection sandbox can only reach GitHub, and the usual
upstream dataset has not started the 2026-27 season. This fills the gap from
the league's own feed and puts the result somewhere the sandbox can read.

Design rules, learned the hard way:

  * Every game is checked against its own published final score before being
    written. A box score that does not foot to the scoreboard is a transcription
    or parsing fault, and a wrong number that looks ordinary is worse than a
    missing one.
  * The script FAILS LOUDLY. If a game that should be final cannot be fetched or
    does not validate, it exits non-zero so the Actions run goes red and you get
    an email, rather than silently committing yesterday's file again.
  * It is idempotent. Re-running it adds nothing and breaks nothing, so a
    retry or a manual run is always safe.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

OUT = os.path.join('data', 'playerlog_2027.parquet')
LOG = os.path.join('data', 'ingest_log.csv')
SEASON_PREFIX = '00226'          # regular season, 2026-27
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/125.0 Safari/537.36')

SCOREBOARD = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'
BOXSCORE = 'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json'
SCHEDULE = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json'


def get(url: str, tries: int = 4) -> dict:
    """Fetch JSON with a browser user agent and a polite backoff."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': UA,
                'Referer': 'https://www.nba.com/',
                'Accept': 'application/json',
            })
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f'could not fetch {url}: {last}')


def eastern_date(offset_days: int = 0) -> str:
    """NBA games are dated by their US Eastern tip-off, not by UTC. A 10:30pm
    Pacific game is still 'yesterday' well into the following UTC day, so using
    UTC here would silently skip the late west-coast games."""
    et = datetime.now(timezone.utc) - timedelta(hours=5)
    return (et + timedelta(days=offset_days)).strftime('%Y-%m-%d')


def games_for(dates: set[str]) -> list[dict]:
    """Every regular-season game on the given US Eastern dates."""
    sched = get(SCHEDULE)
    out = []
    for day in sched['leagueSchedule']['gameDates']:
        d = datetime.strptime(day['gameDate'].split()[0], '%m/%d/%Y').strftime('%Y-%m-%d')
        if d not in dates:
            continue
        for g in day['games']:
            if not str(g['gameId']).startswith(SEASON_PREFIX):
                continue
            out.append(dict(
                gameId=g['gameId'], date=d,
                away=g['awayTeam']['teamTricode'], home=g['homeTeam']['teamTricode'],
                away_pts=g['awayTeam'].get('score'), home_pts=g['homeTeam'].get('score'),
                status=g.get('gameStatus')))        # 3 = final
    return out


def pm(s) -> float:
    """'PT34M12.00S' or '34:12' to decimal minutes."""
    if s is None:
        return 0.0
    s = str(s)
    if s.startswith('PT'):
        m = s[2:].split('M')[0]
        sec = s.split('M')[1].rstrip('S') if 'M' in s else '0'
        try:
            return float(m) + float(sec) / 60
        except ValueError:
            return 0.0
    if ':' in s:
        a, b = s.split(':')[:2]
        try:
            return float(a) + float(b) / 60
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_box(gid: str) -> pd.DataFrame | None:
    """One game's player rows, or None if the feed has not published it."""
    try:
        d = get(BOXSCORE.format(gid=gid))
    except RuntimeError:
        return None
    g = d.get('game')
    if not g:
        return None

    rows = []
    for side in ('homeTeam', 'awayTeam'):
        t = g[side]
        tri = t['teamTricode']
        for p in t.get('players', []):
            st = p.get('statistics') or {}
            mins = pm(st.get('minutes'))
            if mins <= 0:
                continue                             # did not play: no information
            rows.append(dict(
                gameId=gid, teamTricode=tri,
                personId=p.get('personId'),
                firstName=p.get('firstName', ''),
                familyName=p.get('familyName', ''),
                mins=mins,
                points=st.get('points', 0),
                reboundsTotal=st.get('reboundsTotal', 0),
                reboundsOffensive=st.get('reboundsOffensive', 0),
                assists=st.get('assists', 0),
                steals=st.get('steals', 0),
                blocks=st.get('blocks', 0),
                turnovers=st.get('turnovers', 0),
                foulsPersonal=st.get('foulsPersonal', 0),
                fieldGoalsMade=st.get('fieldGoalsMade', 0),
                fieldGoalsAttempted=st.get('fieldGoalsAttempted', 0),
                threePointersMade=st.get('threePointersMade', 0),
                threePointersAttempted=st.get('threePointersAttempted', 0),
                freeThrowsMade=st.get('freeThrowsMade', 0),
                freeThrowsAttempted=st.get('freeThrowsAttempted', 0),
                # the feed marks starters with a position; bench players have none
                position=(p.get('position') or ''),
            ))
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


def main() -> int:
    # yesterday and today in US Eastern: late west-coast games finish after
    # midnight Eastern, so a single day would miss them
    want = {eastern_date(-1), eastern_date(0)}
    print(f'checking US Eastern dates: {sorted(want)}')

    sched = games_for(want)
    finals = [g for g in sched if g['status'] == 3]
    print(f'{len(sched)} games scheduled, {len(finals)} final')
    if not finals:
        print('nothing final yet; nothing to do')
        return 0

    have = set()
    if os.path.exists(OUT):
        have = set(pd.read_parquet(OUT).gameId.astype(str).unique())
    todo = [g for g in finals if g['gameId'] not in have]
    print(f'{len(have)} games already stored, {len(todo)} to add')
    if not todo:
        return 0

    frames, failed, log = [], [], []
    for g in todo:
        box = parse_box(g['gameId'])
        if box is None or box.empty:
            failed.append((g['gameId'], f"{g['away']}@{g['home']}", 'no box score published'))
            continue
        ok = True
        for tri, want_pts in ((g['away'], g['away_pts']), (g['home'], g['home_pts'])):
            got = int(box.loc[box.teamTricode == tri, 'points'].sum())
            if want_pts is None or got != int(want_pts):
                failed.append((g['gameId'], f"{g['away']}@{g['home']}",
                               f'{tri} sums to {got}, scoreboard says {want_pts}'))
                ok = False
        if not ok:
            continue
        frames.append(box)
        log.append(dict(gameId=g['gameId'], date=g['date'],
                        fixture=f"{g['away']}@{g['home']}",
                        score=f"{g['away_pts']}-{g['home_pts']}", players=len(box),
                        ingested=datetime.now(timezone.utc).isoformat(timespec='seconds')))
        print(f"  ok {g['away']}@{g['home']} {g['away_pts']}-{g['home_pts']} "
              f"({len(box)} players)")

    if frames:
        new = derive(pd.concat(frames, ignore_index=True))
        if os.path.exists(OUT):
            new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
            new = new.drop_duplicates(subset=['gameId', 'personId'], keep='last')
        os.makedirs('data', exist_ok=True)
        new.to_parquet(OUT, index=False)
        pd.DataFrame(log).to_csv(
            LOG, mode='a', header=not os.path.exists(LOG), index=False)
        print(f'\nwrote {OUT}: {new.gameId.nunique()} games, {len(new)} player-games')

    if failed:
        print('\nFAILED:', file=sys.stderr)
        for gid, fx, why in failed:
            print(f'  {gid} {fx}: {why}', file=sys.stderr)
        # a game that is final but did not validate is a real fault. Going red
        # here is the point: a silent miss would leave the model running on
        # stale baselines with nobody any the wiser.
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
