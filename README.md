# Valorant Veto Assistant

A machine learning model that helps managers estimate map win probability for Valorant matches, built to support veto decisions in tournament play.

**Input:** two teams and a map.
**Output:** estimated probability that the first team wins that map.

Trained on professional match data from VLR.gg, then tested in a tier 3 environment to see whether patterns learned at professional level transfer down, and whether they help with team success.

## Why

Veto happens under time pressure with incomplete information. Managers currently rely on memory and rough impressions of what an opponent is comfortable on. This replaces the guess with a number.

## v1 scope

Logistic regression implemented in numpy, with gradient descent written from scratch rather than using an ML library. The goal is to understand the mechanics, not to reach maximum accuracy. Benchmarked against a historical-frequency baseline with a three-level fallback (team-map rate, then team overall rate, then 50/50) to handle the large number of team-map pairs with very few games.

## Roadmap

- **v2** — PyTorch, richer features
- **v3** — sequence modelling over match history
- **v4** — web interface

## Data

Scraped from VLR.gg: roughly 7,800 professional matches across all regions and
over two years, giving 18,414 map results and several thousand veto steps. The
scraper caches every page locally so re-runs don't hit the site again.

Two datasets, joined on match ID:

- `vetoData.csv` — one row per veto step (ban, pick, or decider)
- `mapResults.csv` — one row per map played, with scores, attack/defence splits,
  team IDs, match date, and which team picked it

A known limitation: matches older than roughly mid-2024 fail to parse, because
VLR's page structure changed at some point and the scraper's selectors no
longer match. This sets a natural lower bound on the dataset's date range.

Most team-map pairs are sparse — the majority have three games or fewer,
because the scrape sweeps every region and tier rather than a single league.
This is why the baseline falls back to a team's overall win rate, or 50%, when
a specific team-map pair doesn't have enough data (currently a 10-game
threshold) to be trusted.

## Initial findings

| Measure | Rate | n |
|---|---|---|
| Picked maps won by the picking team | 51.8% | 18,414 |
| Team A wins (all maps) | 54.2% | 18,414 |
| Team A wins (decider maps only) | 54.7% | — |

Picking a map is worth about 2 percentage points over a coin flip — a smaller
edge than expected, suggesting professional teams already veto competently.
By the time bans are done, the remaining pool is close to balanced.

Team A (however VLR orders the two teams in a match) wins noticeably more
often than team B across the board — 54.2% overall, not just on deciders. At
n=18,414 this is far outside what chance would produce, so it looks like a
real, structural property of how VLR lists matches (likely related to seeding
or listing order) rather than a skill effect. It should be **excluded as a
model feature** to avoid leaking a spurious signal, and flagged clearly in
any writeup as a property of the data source rather than the game.

These numbers set a realistic ceiling for the model. Map outcomes carry
limited signal, so any claim of high predictive accuracy would more likely
indicate overfitting or leakage than a genuine result.

## Baseline results

A frequency-table baseline (with the fallback chain described above) was
evaluated with a **time-based train/test split** — trained on the earliest 80%
of matches by date, tested on the most recent 20% — to simulate genuine
future prediction rather than leaking outcomes backwards in time.

| | |
|---|---|
| Train / test size | 14,731 / 3,683 maps |
| Baseline accuracy | **55.6%** |

This is the number the v1 logistic regression needs to beat. Given the small
size of the underlying effects (pick edge, side-order edge), a meaningful
improvement over this baseline — rather than a large one — should be
considered a genuine result.