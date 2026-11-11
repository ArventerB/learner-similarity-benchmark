# One rule for building learner ability vectors

Every interaction-log source is turned into learner vectors with the same rule, so that no
view needs a hand-made design. The implementation is `src/lsb/build_views.py`.

## Learner partition

Learners are shuffled once per source (seed 2026).

| Part | Size | Used for |
|---|---|---|
| Reference pool | 20 % of the learners, at most 3,000 | defining the features (which KCs, their prior means) |
| Test set | 40 % of the remainder, at most 1,000 | scoring only; never used to fit anything |
| Candidate pool | the rest | training samples of n learners are drawn from it |

A training size n is run only when the candidate pool holds at least 1.5 n learners.

## Features

1. Count, in the reference pool, how many learners practised each knowledge component (KC)
   in the feature interactions. Keep the d most widely practised KCs, d ∈ {20, 50, 100}.
2. For learner i and KC k, with n_ik attempts and c_ik (partial) successes,

       f_ik = (c_ik + a · p_k) / (n_ik + a),   a = 2,

   where p_k is the reference-pool mean success on KC k. A KC the learner never attempted
   therefore equals p_k.
3. An item tagged with several KCs counts for each of them.

OULAD has no KCs. Its 32 features are registration lead time, number of previous attempts,
studied credits, and log(1 + x) of first-28-day activity: total clicks, active days,
distinct resources, pre-start clicks, clicks per week (4), clicks per activity type (20),
and the number of assessments handed in. No demographic attribute and no score is used.

## Outcomes: four task types

| Task | Sources | Features come from | Outcome |
|---|---|---|---|
| Future performance | ASSISTments 2009 / 2015 / 2017, EdNet, XES3G5M, Slepemapy | the first 70 % of a learner's interactions | success rate on the last 30 % |
| Cross-course transfer | PTADisc cross-course (5 views) | the four other programming courses | mean score in the held-out course |
| Unseen skills | PTADisc single-course (3 views) | a random 70 % of the KCs | mean score on items whose KCs are all in the other 30 % |
| Early warning | OULAD (6 modules) | the first 28 days | fail or withdraw (binary) |

Eligibility: at least 80 interactions (50 for ASSISTments 2009 and 2015) and at least 15
outcome interactions for the temporal views; at least 30 feature-side and 15 outcome-side
interactions for the unseen-skill views; still registered on day 28 for OULAD.

## Preprocessing inside each repetition (training learners only)

1. **Support filter.** A column is dropped when fewer than five training learners depart
   from its most frequent value, that is, when fewer than five of them practised the KC.
2. **Standardisation** with the training mean and standard deviation, clipped at ±5.
3. Covariance estimates and learned metrics are fitted on the standardised training
   matrix. Columns that are constant in the training sample receive zero weight in
   learned metrics.

## Leaky variants kept for the protocol study

`*_randsplit` views draw the outcome interactions at random instead of from the future.
They are used only to measure how much a random split inflates the scores.
