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

## v1.1: pick and map-identity features

Tested whether adding two more features would improve on the v1 model:

- **Pick information** — whether the canonical "team1" picked this map
  (encoded as +1/-1/0 for team1-picked/team2-picked/decider)
- **Map identity** — a one-hot vector across the 12 maps seen in training
  data, so the model can learn map-specific baseline tendencies

Both were built correctly and confirmed to reach the model (feature vector
grew from 4 to 17 columns; map vocabulary built from training data only, to
avoid leaking test-period maps).

| Model | 80/20 split accuracy |
|---|---|
| v1 (4 features) | 55.4% |
| v1.1 (17 features) | 55.5% |

**Result: no meaningful improvement.** The 0.1pp difference is within noise
(the v1 model alone showed a 55.1-56.6% spread across different train/test
splits). Two features grounded in real, separately-measured effects didn't
move the model.

**Why this is a useful negative result, not a failure:**
- Map identity may be largely redundant with the existing map-specific win
  rate features (`rate1`/`rate2`), which already encode "how does this team
  do on this specific map" — a separate "this is Ascent" flag adds little
  once that's already present.
- The picker's edge is small (~2 percentage points, per the pick-rate
  finding above) and may be too weak a signal for a linear model to extract
  when competing against stronger win-rate features.

This suggests the ~55% ceiling isn't a missing-features problem — it's
closer to the actual predictability limit of win-rate-based signals for
this dataset. Moving beyond it likely needs either a non-linear model
capable of interactions the logistic regression can't represent (v2), or
denser per-team-map data (a known limitation, see above), rather than more
features of the same kind.

## v1.2: attack/defence round rates (tested, not adopted)

Added each team's historical attack-side and defence-side round win rate,
built from round counts already present in `mapResults.csv`.

Initial testing showed a large accuracy jump (68%+), but this turned out to
be an evaluation bug — a variable mix-up meant the model was briefly being
tested on its own training data rather than held-out matches. Once fixed:

| Model | Test accuracy |
|---|---|
| v1 (4 features) | 55.1-56.6% |
| v1.1 (+ pick, map identity) | 55.5% |
| v1.2 (+ attack/defence rates) | 55.3% |

No improvement over v1, consistent with v1.1. The trained weights show
attack/defence features contribute real but modest weight (roughly a fifth
of the map win-rate features), which is consistent with this outcome —
a moderate signal that doesn't shift the model's overall accuracy beyond
the existing ~55% ceiling.

**Lesson learned in the process:** the initial 68% result passed three
genuine checks (independent reimplementation of the round-counting logic,
no data leakage, stability across three train/test splits) before a fourth
check — comparing train and test accuracy — accidentally used the same
data for both, producing a false confirmation. Re-running with the correct
split showed the true result. Kept here as a reminder to verify *which
data* a check is actually using, not just whether the check passes.