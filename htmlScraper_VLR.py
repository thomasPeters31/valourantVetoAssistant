# --- Where this fits ---
# This is the data-collection stage of the pipeline:
#   htmlScraper_VLR.py  -> scrapes VLR.gg and writes mapResults.csv (this file)
#   analysis.py          -> reads mapResults.csv, computes win-rate stats,
#                            and exposes buildFrequencyTable()/predictWinRate()
#   features.py           -> reads mapResults.csv + analysis.py's helpers to
#                            build a numpy feature/label dataset for modeling
#
# Run this file directly (`python htmlScraper_VLR.py`) to (re)build
# mapResults.csv from scratch — see rebuildMapResultsFromRegion() at the
# bottom. Every match page fetched is cached under cache/<id>.html, so
# re-running only makes network requests for matches that aren't cached yet.

import requests
import os
from bs4 import BeautifulSoup
import time
import csv
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}


def getVetoSequence(html):
    # Parses the map pick/ban sequence out of a match page's HTML.

    soup = BeautifulSoup(html, "html.parser")
    note = soup.select_one("div.match-header-note")

    # Not every match page has a veto note (e.g. bo1s, or ones VLR hasn't filled in).
    if note is None:
        return []

    # VLR.gg puts the whole veto in one plain-text blob rather than tagged
    # elements, e.g. "SEN ban Icebox; ENVY ban Split; ...; Sunset remains".
    # get_text() strips the tags and normalizes whitespace to single spaces.
    text = note.get_text(separator=" ", strip=True)

    # Each veto step is separated by a semicolon.
    segments = text.split(";")

    # Splitting on ";" leaves leading/trailing spaces on each piece; trim them.
    cleanSegments = [segment.strip() for segment in segments]

    veto = []

    for segment in cleanSegments:
        # "SEN ban Icebox" -> ["SEN", "ban", "Icebox"]
        parts = segment.split()

        if len(parts) >= 2 and parts[1] in ("ban", "pick"):
            # Normal step: team, action, then the map (joined back in case
            # a map name is ever multi-word).
            entry = {"team": parts[0], "action": parts[1], "map": " ".join(parts[2:])}
        elif segment.endswith("remains"):
            # The decider map is never picked, e.g. "Sunset remains" —
            # no team or action attached to it.
            entry = {"team": None, "action": "decider", "map": parts[0]}
        else:
            # Doesn't match either known shape, meaning VLR changed its
            # format or this note is worded differently. Flag it instead
            # of silently dropping a veto step.
            print(f"Unexpected segment format: {segment}")
            continue

        veto.append(entry)

    return veto


def getHtml(url, cacheName, useCache=True):
    # Fetches a page's HTML, caching it on disk under cacheName so repeated
    # runs don't re-request pages we've already scraped. Every other
    # function in this file that hits the network goes through here.
    os.makedirs("cache", exist_ok=True)
    path = f"cache/{cacheName}.html"

    if useCache and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()

    print(f"Fetching {cacheName}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.RequestException as e:
        # Network-level failure (timeout, DNS, connection reset, etc.) —
        # give up on this page and let the caller deal with "".
        print(f"Request failed for {url}: {e}")
        return ""
    # Small delay so a batch of uncached requests doesn't hammer VLR.gg.
    time.sleep(3)

    if response.status_code != 200:
        # e.g. 404/429/5xx — don't cache a bad response as if it were real data.
        print(f"Failed to fetch {url}: {response.status_code}")
        return ""

    with open(path, "w") as f:
        f.write(response.text)

    return response.text


def getMatchIDs(pageNumber):
    # Scrapes one page of VLR's GLOBAL results listing for the match IDs on it.
    # Kept around for the old sweep-everything approach (see rebuildMapResults
    # below) — superseded as the default by the event-based discovery further
    # down, but still useful if you ever want an unfiltered top-up.
    url = f"https://www.vlr.gg/matches/results?page={pageNumber}"
    # useCache=False: this listing changes constantly as new results land,
    # so a cached copy would quickly go stale.
    html = getHtml(url, f"results_page_{pageNumber}", useCache=False)
    soup = BeautifulSoup(html, "html.parser")

    matchIDs = []
    for card in soup.select("a.match-item"):
        # Each match card carries small tag chips (e.g. "Map", "VOD");
        # only "Map"-tagged matches have completed map data to scrape.
        tags = card.select("div.match-item-vod div.wf-tag")
        tagTexts = [t.get_text(strip=True) for t in tags]

        if "Map" not in tagTexts:
            continue  # Skip matches that don't have a map tag (e.g., unplayed matches)

        matchIDs.append(card['href'].split('/')[1])  # Extract match ID from URL

    return matchIDs


def getAllMatchIDs(numPages):
    # Walks the GLOBAL results listing page by page (1-indexed) and flattens
    # every match ID found into one list. Sweeps every region/tier — this is
    # the source of the team-map sparsity problem the event-based approach
    # below is meant to fix.
    allIDs = []
    for page in range(1, numPages + 1):
        ids = getMatchIDs(page)
        allIDs.extend(ids)
    return allIDs


def getEventIDs(region, tier, maxPages=10, minYear=2023):
    eventIDs = []

    for page in range(1, maxPages + 1):
        url = f"https://www.vlr.gg/events/?region={region}&tier={tier}&page={page}"
        html = getHtml(url, f"events_{region}_{tier}_page_{page}", useCache=False)
        soup = BeautifulSoup(html, "html.parser")

        cards = soup.select("a.event-item")
        if not cards:
            break

        for card in cards:
            parts = card['href'].split('/')
            eventID = parts[2]
            slug = parts[3]

            yearMatch = re.search(r"20\d{2}", slug)
            if not yearMatch or int(yearMatch.group()) < minYear:
                continue  # pre-franchise era: different team pool, different page template

            eventIDs.append((eventID, slug))

    return eventIDs


def getMatchIDsFromEvent(eventID, slug):
    # Scrapes one event's match listing for the match IDs on it. Same
    # match-card / "Map" tag filter as getMatchIDs, pointed at a single
    # event instead of the global feed.
    url = f"https://www.vlr.gg/event/matches/{eventID}/{slug}/"
    # useCache=True: a completed event's match list is finished changing,
    # unlike the live global results feed above.
    html = getHtml(url, f"event_{eventID}_matches", useCache=True)
    soup = BeautifulSoup(html, "html.parser")

    matchIDs = []
    for card in soup.select("a.match-item"):
        tags = card.select("div.match-item-vod div.wf-tag")
        tagTexts = [t.get_text(strip=True) for t in tags]

        if "Map" not in tagTexts:
            continue

        matchIDs.append(card['href'].split('/')[1])

    return matchIDs


def getAllMatchIDsFromRegion(region, tier):
    # Discovers every match ID across every VCT-tier event in one region.
    # This is the event-based replacement for getAllMatchIDs.
    events = getEventIDs(region, tier)
    print(f"Found {len(events)} events")

    allIDs = []
    for eventID, slug in events:
        ids = getMatchIDsFromEvent(eventID, slug)
        print(f"{slug}: {len(ids)} matches")
        allIDs.extend(ids)

    return allIDs


def getMapResults(html):
    # Parses the per-map scoreline/side-score breakdown from a match page
    # (as opposed to getVetoSequence, which parses the pick/ban note).
    soup = BeautifulSoup(html, "html.parser")
    games = soup.select("div.vm-stats-game")

    results = []

    for game in games:
        gameID = game.get("data-game-id")

        if gameID == "all":
            continue  # Skip the aggregate stats block

        # The map name is the first text node inside this span, ahead of
        # any icon/badge markup (like the "picked" tag handled below).
        span = game.select_one("div.map span")
        mapName = span.contents[0].strip()
        pickedSpan = span.select_one("span.picked")

        # VLR tags whichever map a team picked with a "picked" span whose
        # class marks which side (mod-1 = team A, mod-2 = team B) chose it.
        # A map with no such span is the decider — neither team picked it.
        if pickedSpan is None:
            pickedBy = None          # decider — neither team picked it
        elif "mod-1" in pickedSpan.get("class", []):
            pickedBy = "a"
        else:
            pickedBy = "b"

        header = game.select_one("div.vm-stats-game-header")
        teams = header.select("div.team")

        # VLR always lists exactly two teams per map header, in match order.
        teamA = parseTeam(teams[0])
        teamB = parseTeam(teams[1])

        entry = {
            "map": mapName,
            "pickedBy": pickedBy,
            "team_a": teamA["name"],
            "score_a": teamA["score"],
            "attack_a": teamA["attack"],
            "defence_a": teamA["defence"],
            "team_b": teamB["name"],
            "score_b": teamB["score"],
            "attack_b": teamB["attack"],
            "defence_b": teamB["defence"],
            "winner": teamA["name"] if teamA["won"] else teamB["name"],
        }

        results.append(entry)

    return results


def getMatchTeams(html):
    # Pulls each team's ID and display name from the match header (as
    # opposed to parseTeam, which reads a team's per-map score block).
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("a.match-header-link")

    teams = []
    for link in links:
        teamID = link.get("href").split("/")[2]

        # The name div sometimes holds a second div with a shorter display name
        # alongside the registered one, so take just the first text node.
        nameDiv = link.select_one("div.wf-title-med")
        name = nameDiv.contents[0].strip()

        teams.append({"teamID": teamID, "name": name})

    return teams


def parseTeam(teamDiv):
    # Pulls one team's name/score/side-scores out of a vm-stats-game-header
    # "div.team" block.
    scoreElement = teamDiv.select_one("div.score")

    return {
        "name": teamDiv.select_one("div.team-name").get_text(strip=True),
        "score": int(scoreElement.get_text(strip=True)),
        # VLR marks the winning team's score element with a "mod-win" class.
        "won": "mod-win" in scoreElement.get("class", []),
        # "mod-t"/"mod-ct" are the rounds won on attack vs. defence.
        "attack": int(teamDiv.select_one("span.mod-t").get_text(strip=True)),
        "defence": int(teamDiv.select_one("span.mod-ct").get_text(strip=True)),
    }


def getMatchDate(html):
    # Reads the match's UTC timestamp from its header, used by analysis.py's
    # baseLineEvaluation() to sort matches chronologically for a time-based
    # train/test split.
    soup = BeautifulSoup(html, "html.parser")
    dateElement = soup.select_one("div.moment-tz-convert")

    if dateElement is None:
        return None

    return dateElement.get("data-utc-ts")


def buildRowsFromMatchIDs(matchIDs):
    # Shared by both rebuild paths below: given a list of match IDs, fetch
    # each match page and turn it into per-map rows. This is the one place
    # that logic lives, so the global sweep and the event-based scrape can't
    # drift apart from each other the way append/rebuild did before.
    rows = []

    for matchID in matchIDs:
        html = getHtml(f"https://www.vlr.gg/{matchID}", f"match_{matchID}")
        try:
            maps = getMapResults(html)
            headerTeams = getMatchTeams(html)
            date = getMatchDate(html)

            if len(headerTeams) != 2:
                print(f"SKIP {matchID}: found {len(headerTeams)} header teams")
                continue

            for entry in maps:
                entry["matchID"] = matchID
                entry["teamID_a"] = headerTeams[0]["teamID"]
                entry["teamID_b"] = headerTeams[1]["teamID"]
                entry["date"] = date
                rows.append(entry)

        except Exception as e:
            # Keep going on a bad/unexpected page rather than aborting the
            # whole rebuild over one match.
            print(f"FAILED {matchID}: {type(e).__name__}: {e}")

    return rows


def writeMapResults(rows):
    # Overwrites mapResults.csv with exactly these rows. Shared write path
    # for both rebuild functions below.
    with open("mapResults.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "matchID", "map", "pickedBy", "date",
            "teamID_a", "team_a", "score_a", "attack_a", "defence_a",
            "teamID_b", "team_b", "score_b", "attack_b", "defence_b",
            "winner",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to mapResults.csv")


def getMoreData():
    # Incremental alternative: only fetches matches not already present in
    # mapResults.csv, appending rather than rebuilding. Sweeps the global
    # feed (pages 1-160) — kept for topping up between full rebuilds.
    with open("mapResults.csv") as f:
        existingIDs = {row["matchID"] for row in csv.DictReader(f)}

    print(f"Already have {len(existingIDs)} matches")

    allIDs = getAllMatchIDs(160)
    newIDs = [mid for mid in allIDs if mid not in existingIDs]

    print(f"Found {len(newIDs)} new matches to scrape")

    newRows = buildRowsFromMatchIDs(newIDs)

    with open("mapResults.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "matchID", "map", "pickedBy",
            "teamID_a", "team_a", "score_a", "attack_a", "defence_a",
            "teamID_b", "team_b", "score_b", "attack_b", "defence_b",
            "winner", "date"
        ])
        writer.writerows(newRows)   # no writeheader() — the header's already there

    print(f"Appended {len(newRows)} rows to mapResults.csv")


def rebuildMapResults(numPages):
    # Old approach: sweep every region/tier via the global results feed.
    # Fully overwrites mapResults.csv.
    with open("mapResults.csv") as f:
        existingIDs = {row["matchID"] for row in csv.DictReader(f)}

    freshIDs = set(getAllMatchIDs(numPages))
    allIDs = sorted(existingIDs | freshIDs)

    print(f"Rebuilding from {len(allIDs)} total matches")

    rows = buildRowsFromMatchIDs(allIDs)
    writeMapResults(rows)


def rebuildMapResultsFromRegion(region, tier):
    # New approach: only matches from VCT-tier events in one region.
    # Denser per-team-map data, at the cost of a smaller total dataset.
    # Fully overwrites mapResults.csv.
    matchIDs = sorted(set(getAllMatchIDsFromRegion(region, tier)))

    print(f"Rebuilding from {len(matchIDs)} total matches")

    rows = buildRowsFromMatchIDs(matchIDs)
    writeMapResults(rows)


if __name__ == "__main__":
    # Region-based build: VCT Americas only (region=26, tier=60).
    # Swap the region code to expand to another league later —
    # everything else in the pipeline is region-agnostic.
    rebuildMapResultsFromRegion(26, 60)