# Valorant Veto Assistant

A machine learning model that helps managers estimate map win probability for Valorant matches, built to support veto decisions in tournament play.

**Input:** two teams and a map.
**Output:** estimated probability that the first team wins that map.

Trained on professional match data from VLR.gg, then tested in a tier 3 environment to see whether patterns learned at professional level transfer down, and whether they help with team success.

## Why

Veto happens under time pressure with incomplete information. Managers currently rely on memory and rough impressions of what an opponent is comfortable on. This replaces the guess with a number.

## Scope

Two models were built and compared: a historical-frequency baseline, and a
logistic regression implemented from scratch in numpy (gradient descent
hand-written, not from a library) to understand the mechanics rather than
just call `.fit()`. A PyTorch model was added later specifically to test
whether a more expressive architecture could do better.

## Data

Scraped from VLR.gg: roughly 7,800 professional matches across all regions
and over two years, giving 18,414 map results. The scraper caches every
page locally so re-runs don't hit the site again.

Two datasets, joined on match ID:

- `vetoData.csv` — one row per veto step (ban, pick, or decider)
- `mapResults.csv` — one row per map played, with scores, attack/defence
  splits, team IDs, match date, and which team picked it

**Known limitation:** matches older than roughly mid-2024 fail to parse,
because VLR's page structure changed at some point and the scraper's
selectors no longer match.

**Known limitation, and the central constraint on this whole project:**
most team-map pairs are sparse — the majority have three games or fewer,
because the scrape sweeps every region and tier rather than a single
league. This is why every model here falls back to a team's overall win
rate, or 50%, when a specific team-map pair doesn't have enough data
(currently a 10-game threshold) to be trusted. As the results below show,
this sparsity turned out to be the actual limiting factor on accuracy —
more so than model choice or feature choice.

## Initial findings

| Measure | Rate | n |
|---|---|---|
| Picked maps won by the picking team | 51.8% | 18,414 |
| Team A wins (all maps) | 54.2% | 18,414 |
| Team A wins (decider maps only) | 54.7% | 18,414 |

Picking a map is worth about 2 percentage points over a coin flip —
smaller than expected, suggesting professional teams already veto
competently. Team A (however VLR happens to order the two teams in a
match) wins noticeably more often than team B, a structural artefact of
listing order rather than a skill effect. Every model below **canonically
re-orders each match's two teams by ID** to remove this leak, rather than
trusting VLR's listing order.

These numbers set a realistic ceiling for the model. Map outcomes carry
limited signal, so any claim of high predictive accuracy would more likely
indicate overfitting or leakage than a genuine result.

## Results: a systematic search for what actually helps

Every model below was evaluated with a **time-based train/test split**
(trained on the earliest matches by date, tested on the most recent) to
simulate genuine future prediction. The frequency-table baseline and the
logistic regression were both checked for stability across three different
split proportions (75/25, 80/20, 90/10), not just one.

| Model | Features | Test accuracy |
|---|---|---|
| Frequency baseline | Team-map rate, team overall rate (3-level fallback) | 55.6% (80/20) |
| v1: logistic regression | Same as baseline, numpy from scratch | 55.1–56.6% across splits |
| v1.1: + pick, map identity | + which team picked, one-hot map | 55.5% — no improvement |
| v1.2: + attack/defence rounds | + per-team attack/defence round win rate | 55.3% — no improvement |
| v2: PyTorch MLP | Same features as v1.2, one hidden layer (16 units, ReLU) | 55.6% — no improvement |

**Every model, regardless of features or architecture, lands in the same
55–57% band.**

### What was ruled out, and how

- **Pick and map-identity features** (v1.1) were added and verified to
  reach the model correctly (feature vector grew from 4 to 17 columns,
  confirmed via shape check) but produced no measurable improvement.
- **Attack/defence round rates** (v1.2) initially appeared to jump
  accuracy to ~68%. This was investigated rather than accepted at face
  value: the round-counting logic was verified against an independent
  from-scratch reimplementation (exact match), leakage was ruled out by
  confirming a team's historical rate is built entirely from other
  matches, and the result was checked across three splits. It held up
  every time — right up until a train/test accuracy comparison, intended
  as a final overfitting check, revealed the true cause: a variable
  mix-up meant the model had briefly been evaluated against its own
  training data rather than held-out matches. The corrected evaluation
  showed 55.3% — in line with everything else. Kept in this writeup
  because catching it was as instructive as the result itself: a check
  that passes is only meaningful if you're certain what data it actually
  used.
- **Model capacity** (v2) was tested by moving from a linear model to a
  small neural network capable of learning feature interactions a linear
  model can't represent. It converged to 55.6% — the same band. This
  points specifically at *data density*, not *model expressiveness*, as
  the limiting factor: a more capable model found nothing extra to
  exploit in the features available.

### Conclusion

Four different levers were tried — richer features (pick, map identity),
finer-grained features (round-level data), and a more expressive model
(neural network) — and none moved accuracy beyond the ~55% ceiling
established by the simplest baseline. Combined with the known sparsity of
per-team-map data (most pairs have 3 games or fewer), the evidence points
toward **data density, not modelling choice, as the actual constraint**.

## Roadmap

- **v3** — scrape full seasons of specific VCT regional leagues (Americas,
  EMEA, Pacific, China) rather than sweeping every region and tier, to
  produce denser per-team-map data. This is the direct response to the
  finding above, and the most promising remaining lever.
- **v4** — sequence modelling over match history (recent form, streaks)
  rather than static career-average rates, once v3 provides enough
  per-team data to make sequences meaningful.
- **v5** — web interface for the manager to query a matchup directly.