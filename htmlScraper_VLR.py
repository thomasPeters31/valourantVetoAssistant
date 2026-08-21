import requests
import os
from bs4 import BeautifulSoup
import time

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


def getHtml(url, cacheName):
    # Cache each match's HTML on disk by ID so re-running the script doesn't
    # re-fetch pages we've already scraped.
    os.makedirs("cache", exist_ok=True)
    path = f"cache/{cacheName}.html"

    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return ""

    # Small delay so a batch of uncached match_ids doesn't hammer VLR.gg.
    time.sleep(1)

    with open(path, "w") as f:
        f.write(response.text)

    return response.text

def getMatchIDs(pageNumber):
    url = f"https://www.vlr.gg/matches/results?page={pageNumber}"
    html = getHtml(url, f"results_page_{pageNumber}")
    soup = BeautifulSoup(html, "html.parser")
    
    matchIDs = []
    for card in soup.select("a.match-item"):
        tags = card.select("div.match-item-vod div.wf-tag")
        tagTexts = [t.getText(strip=True) for t in tags]
        
        if "Map" not in tagTexts:
            continue  # Skip matches that don't have a map tag (e.g., unplayed matches)
        
        matchIDs.append(card['href'].split('/')[1])  # Extract match ID from URL

    return matchIDs

if __name__ == "__main__":
    ids = getMatchIDs(1)
    
    for match_id in ids:
        html = getHtml(f"https://www.vlr.gg/{match_id}", f"match_{match_id}")
        veto = getVetoSequence(html)
        print(match_id, len(veto))
