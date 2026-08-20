import requests
import os
from bs4 import BeautifulSoup

url = "https://www.vlr.gg/729747/sentinels-vs-envy-vct-2026-americas-stage-2-ubqf"
headers = {"User-Agent": "Mozilla/5.0"}

# Cache the page locally so repeated runs don't hammer VLR.gg with requests.
# Only fetch over the network the first time; every run after that reads match.html.
if not os.path.exists("match.html"):
    response = requests.get(url, headers=headers)
    print(response.status_code)
    with open("match.html", "w") as f:
        f.write(response.text)

with open("match.html", "r") as f:
    htmlContent = f.read()

soup = BeautifulSoup(htmlContent, "html.parser")

# VLR.gg puts the whole veto as one plain-text note, not individual tagged
# elements, e.g. <div class="match-header-note">SEN ban Icebox; ... ; Bind remains</div>
note = soup.select_one("div.match-header-note")

# Collapse the note down to a single space-separated string of just its text,
# stripping out the surrounding tags/whitespace.
text = note.get_text(separator=" ", strip=True)

# Each veto step is separated by a semicolon, e.g. "SEN ban Icebox".
segments = text.split(";")

# The raw segments have leading/trailing whitespace (from the split); trim it.
cleanSegments = [segment.strip() for segment in segments]

veto = []

for segment in cleanSegments:
    # Break "SEN ban Icebox" into ["SEN", "ban", "Icebox"].
    parts = segment.split()

    if len(parts) >= 2 and parts[1] in ("ban", "pick"):
        # Normal veto step: team, action, and (possibly multi-word) map name.
        entry = {"team": parts[0], "action": parts[1], "map": " ".join(parts[2:])}
    elif segment.endswith("remains"):
        # The decider map isn't picked by either team, e.g. "Bind remains" —
        # there's no team or action, just the leftover map.
        entry = {"team": None, "action": "decider", "map": parts[0]}
    else:
        # Anything that doesn't match either shape means VLR.gg's format
        # changed or this page is structured differently — surface it
        # instead of silently dropping data.
        print(f"Unexpected segment format: {segment}")
        continue

    veto.append(entry)

for entry in veto:
    print(entry)
