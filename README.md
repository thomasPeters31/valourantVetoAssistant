# Valorant Veto Assistant

A machine learning model that helps managers estimate map win probability for Valorant matches, built to support veto decisions in tournament play.

**Input:** two teams and a map.
**Output:** estimated probability that the first team wins that map.

Trained on professional match data from VLR.gg, then tested in a tier 3 environment to see whether patterns learned at professional level transfer down, and whether they help with team success.

## Why

Veto happens under time pressure with incomplete information. Managers currently rely on memory and rough impressions of what an opponent is comfortable on. This replaces the guess with a number.

## v1 scope

Logistic regression implemented in numpy, with gradient descent written from scratch rather than using an ML library. The goal is to understand the mechanics, not to reach maximum accuracy. Benchmarked against a simple historical-frequency baseline.

## Roadmap

- **v2** — PyTorch, richer features
- **v3** — sequence modelling over match history
- **v4** — web interface

## Data

Scraped from VLR.gg: roughly 600 professional matches across all regions, giving
1,574 map results and 4,254 veto steps. The scraper caches every page locally so
re-runs don't hit the site again.

Two datasets, joined on match ID:

- `vetoData.csv` — one row per veto step (ban, pick, or decider)
- `mapResults.csv` — one row per map played, with scores, attack/defence splits,
  team IDs, and which team picked it

## Initial findings

| Measure | Rate | n |
|---|---|---|
| Picked maps won by the picking team | 52.7% | 1,326 |
| Team A wins (all maps) | 52.0% | 1,574 |
| Team A wins (decider maps only) | 55.2% | 248 |

Picking a map is worth about 2.7 percentage points over a coin flip. That is a
smaller edge than expected and suggests professional teams already veto
competently — by the time bans are done, the remaining pool is close to balanced.

The decider figure sits 5 points above even, but at n=248 that is within the
range chance would produce and should not be treated as a real effect until the
sample is larger.

These numbers set a realistic ceiling for the model. Map outcomes carry limited
signal, so any claim of high predictive accuracy would more likely indicate
overfitting or leakage than a genuine result.