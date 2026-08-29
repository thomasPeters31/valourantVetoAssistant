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

- **v1.1** — add pick information and map identity as model features
- **v2** — PyTorch, richer features
- **v3** — sequence modelling over match history
- **v4** — web interface

## Data

Scraped from VLR.gg: roughly 7,800 professional matches across all regions and
over two years, giving 18,414 map results. The scraper caches every page
locally so re-runs don't hit the site again.

Two datasets, joined on match ID:

- `vetoData.csv` — one row per veto step (ban, pick, or decider)
- `mapResults.csv` — one row per map played, with scores, attack/defence
  splits, team IDs, match date, and which team picked it

**Known limitation:** matches older than roughly mid-2024 fail to parse,
because VLR's page structure changed at some point and the scraper's
selectors no longer match. This sets a natural lower bound on the dataset's
date range.

**Known limitation:** most team-map pairs are sparse — the majority have
three games or fewer, because the scrape sweeps every region and tier rather
than a single league. This is why the baseline (and the model's features)
fall back to a team's overall win rate, or 50%, when a specific team-map
pair doesn't have enough data (currently a 10-game threshold) to be trusted.

## Initial findings

| Measure | Rate | n |
|---|---|---|
| Picked maps won by the picking team | 51.8% | 18,414 |
| Team A wins (all maps) | 54.2% | 18,414 |
| Team A wins (decider maps only) | 54.7% | 18,414 |

Picking a map is worth about 2 percentage points over a coin flip — a
smaller edge than expected, suggesting professional teams already veto
competently. By the time bans are done, the remaining pool is close to
balanced.

Team A (however VLR orders the two teams in a match) wins noticeably more
often than team B across the board, not just on deciders. At n=18,414 this
is far outside what chance would produce, so it looks like a real,
structural property of how VLR lists matches rather than a skill effect. It
was **excluded as a model feature** to avoid leaking this into predictions —
handled by canonically re-ordering each match's two teams (by team ID)
before building features, rather than trusting VLR's listing order.

These numbers set a realistic ceiling for the model. Map outcomes carry
limited signal, so any claim of high predictive accuracy would more likely
indicate overfitting or leakage than a genuine result.

## Baseline results

A frequency-table baseline (with the fallback chain described above) was
evaluated with a **time-based train/test split** — trained on the earliest
80% of matches by date, tested on the most recent 20% — to simulate genuine
future prediction rather than leaking outcomes backwards in time.

| | |
|---|---|
| Train / test size | 14,731 / 3,683 maps |
| Baseline accuracy | **55.6%** |

## v1 model results

Features per row: each team's map-specific win rate (from the same
fallback-chain frequency table as the baseline) and each team's overall win
rate, with team order canonicalised by ID to remove the listing-order bias
described above.

Logistic regression, trained from scratch with batch gradient descent
(learning rate 0.1, 5,000 epochs — loss converges and flattens by roughly
epoch 4,500).

Evaluated with the same time-based split approach, tested across three
different train/test proportions to check the result wasn't a fluke of one
particular cut:

| Split | Test accuracy |
|---|---|
| 75% / 25% | 55.1% |
| 80% / 20% | 55.4% |
| 90% / 10% | 56.6% |

Accuracy is stable in the 55-57% range across all three splits, consistent
with the frequency baseline's 55.6%. The 90/10 split's higher figure is most
likely down to its smaller test set (~1,840 rows vs ~3,700-4,600) being
noisier, rather than a genuine improvement from more training data.

**Why the model doesn't beat the baseline:** the model's four features are
built from the same frequency-table logic as the baseline itself — two are
literally the baseline's core per-team-map rates, the other two are the
same team's overall rate. The model is essentially learning a weighted,
smoothed version of information the baseline already uses directly, so a
near-identical result is expected rather than a failure. It suggests these
four features capture most of the extractable signal available from win-
rate history alone — moving beyond ~55% will need genuinely new
information, not more training or model complexity on the same inputs.

**What v1.1 will add**, in order of expected impact:
- Which team picked the map (measured separately at ~52% picker win rate,
  not yet in the feature vector)
- Map identity itself (the model currently has no signal for *which* map
  is being played)
- Denser data per team-map pair, most likely by scraping full seasons of
  specific leagues rather than sweeping every region and tier