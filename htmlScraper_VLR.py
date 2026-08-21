import requests
import os
from bs4 import BeautifulSoup
import time
import csv

HEADERS = {"User-Agent": "Mozilla/5.0"}


def getVetoSequence(html):

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
    # Cache each match's HTML on disk by ID so re-running the script doesn't
    # re-fetch pages we've already scraped.
    os.makedirs("cache", exist_ok=True)
    path = f"cache/{cacheName}.html"

    if useCache and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    
    print(f"Fetching {cacheName}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return ""
    # Small delay so a batch of uncached match_ids doesn't hammer VLR.gg.
    time.sleep(3)
    
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return ""

    with open(path, "w") as f:
        f.write(response.text)

    return response.text

def getMatchIDs(pageNumber):
    url = f"https://www.vlr.gg/matches/results?page={pageNumber}"
    html = getHtml(url, f"results_page_{pageNumber}", useCache=False)
    soup = BeautifulSoup(html, "html.parser")
    
    matchIDs = []
    for card in soup.select("a.match-item"):
        tags = card.select("div.match-item-vod div.wf-tag")
        tagTexts = [t.get_text(strip=True) for t in tags]
        
        if "Map" not in tagTexts:
            continue  # Skip matches that don't have a map tag (e.g., unplayed matches)
        
        matchIDs.append(card['href'].split('/')[1])  # Extract match ID from URL

    return matchIDs

def getAllMatchIDs(numPages):
    allIDs = []
    for page in range(1, numPages + 1):
        ids = getMatchIDs(page)
        print(f"Page {page}: {len(ids)} matches")
        allIDs.extend(ids)
    return allIDs

if __name__ == "__main__":
    ids = getAllMatchIDs(20)
    
    allRows = []
    
    for matchID in ids:
        html = getHtml(f"https://www.vlr.gg/{matchID}", f"match_{matchID}")
        veto = getVetoSequence(html)
        
        for entry in veto:
            entry["matchID"] = matchID
            allRows.append(entry)
        
        print(matchID, len(veto))
        
    with open("vetoData.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["matchID", "team", "action", "map"])
        writer.writeheader()
        writer.writerows(allRows)
    print(f"Total veto entries: {len(allRows)}, to vetoData.csv")
