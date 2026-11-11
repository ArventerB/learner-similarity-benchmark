### 5.1 What the benchmark shows

The question that motivated this study was whether a covariance-aware or learned metric should replace the simple measures used for similar-learner retrieval. On eight public sources and for cohorts of 50 to 500 learners, the answer is no. The cosine similarity on standardised ability vectors is the best or co-best retrieval measure at every training size. Every form of whitening (the Mahalanobis distance with the sample covariance, with two shrinkage estimators, and truncated PCA whitening) falls to the level of predicting the training mean. Seven learned metrics, including an oracle over them and thirteen tuned configurations, reach cosine at the largest sample size and do not exceed it.

The result is not that covariance is hard to estimate with few learners. That part of the intuition is correct: shrinkage does improve on the sample covariance, and more at n = 50 than at n = 500. The result is that the quantity being estimated is the wrong target for retrieval on these data. A learner's ability vector has a dominant common factor, the outcomes we care about are driven by it, and whitening is precisely the operation that removes its dominance. The loss from whitening grows with n, tracks the signal–variance alignment across cells with a rank correlation of {{mechanism/d_maha~sva/0|.2f}}, and disappears step by step along the path from the Mahalanobis to the Euclidean distance.

This also explains why learned metrics struggle. A learned linear metric must discover from n outcomes that the high-variance directions matter, which the Euclidean and cosine measures assume for free. With n/d between 0.5 and 25 it has too few outcomes to do better than that assumption, and without a rank constraint it spends its parameters on low-variance directions. The one configuration change that helped, a rank-10 constraint, works by forbidding exactly that.

### 5.2 Relation to earlier work

The systems reviewed in Section 2.1 chose their similarity without comparison. Our results support the choice of a cosine similarity in ReAL (Lv et al., 2025), with the caveat that the benefit depended entirely on standardising the vector first, and that we did not test mastery vectors from a deep knowledge-tracing model. They do not support the use of a Mahalanobis distance as a default: HISE-KT (Duan et al., 2025) applies it to five dense pair features, a setting closer to our OULAD views, where the difference between the Ledoit–Wolf Mahalanobis and the Euclidean distance was smallest ({{by_task/d_maha/warning|.3f}}) but still negative. Whether the Mahalanobis distance helps there is an empirical question that an ablation against the Euclidean distance would settle.

The pattern matches what careful re-evaluations found elsewhere: under a common protocol, elaborate methods often fail to beat a well-configured simple one (Musgrave et al., 2020; Liu et al., 2022). It also adds a positive finding to those negative ones: a diagnostic that says why, and that can be computed on a new data set before any metric is fitted.

### 5.3 Practical guidance

For a practitioner with n ≤ 500 learners and an ability vector of 20 to 100 features, the results translate into the following steps.

1. **Preprocess on the training cohort only.** Drop features that fewer than five training learners have data for, then standardise. Low-support features otherwise dominate every distance.
2. **Use the cosine similarity on the standardised vector as the default.** Its regret against the best measure of each cell was {{selector/always Cosine|.3f}} R² on average, the lowest of all strategies.
3. **Do not whiten.** Compute the signal–variance alignment on the training cohort; if it exceeds 1, as it did in every cell here, a Mahalanobis distance will hurt, with or without shrinkage.
4. **From about 200 learners, compare label-free measures by leave-one-out inside the cohort.** Below that, the comparison is too noisy to beat the default. Never include a learned metric in such a comparison unless it is refitted inside every fold.
5. **Treat a learned metric as optional.** If one is used, constrain its rank and expect parity with cosine, at a fitting cost of seconds per cohort instead of milliseconds.
6. **If the outcome is specific to one skill, check the Euclidean distance or a PCA projection as well.** Cosine matches mainly on general ability, and lost its advantage for skills that general ability did not predict.
7. **If only a prediction is needed, fit a regularised linear model.** From 100 learners it predicts better than any retrieval measure. Retrieval is justified when the neighbours themselves are the product: as context for a language model, as peers to recommend, or as an explanation.
8. **Evaluate with learner-disjoint and temporal splits, and test on another course if the system will be used on one.** A random outcome split inflated R² by about half, and training on other modules removed all calibrated skill.

### 5.4 Limitations

**Ability vectors.** The vectors are smoothed per-skill success rates. They are the transparent analogue of a mastery vector, but they are sparser than the output of a deep knowledge-tracing model, which smooths across skills. A denser, model-based vector may have a weaker common factor after standardisation, and the ranking could differ. The OULAD views, which are dense, showed the same direction of effects at a much smaller size.

**Outcomes.** Four of the five tasks predict an overall performance, which favours measures that match on general ability. The skill-specific task narrows the advantage of cosine and should be extended to more sources and to richer targets such as the response to a single item. The temporal outcomes are raw success rates and are not adjusted for item difficulty, which adaptive systems vary with ability.

**Sample sizes and dimensions.** The grid stops at 500 learners and 100 features. Learned metrics were still improving at n = 500, and with thousands of learners they may overtake the fixed measures. Our conclusions are about the small-cohort regime.

**Metrics.** All measures are linear. Nonlinear and deep metric learning, kernel methods and graph-based similarity were not tested, nor were composite measures that add structural information from a knowledge graph.

**Hyper-parameters.** Learned metrics ran with fixed standard settings, and the sensitivity study covers three learners on a subset of the cells. A full nested tuning could move individual numbers; the hindsight-optimal configuration in Table A12 bounds what it could gain at n ≤ 500.

**Independence and coverage.** The {{n_views}} views come from eight sources; five views share their learners. Source-level tests have little power, and we rely on the consistency of direction across sources. No source is a university transcript, none is from Vietnam, and all are in mathematics, language, geography or computing.

**Mechanism.** The signal–variance alignment exceeded 1 in every cell, so its prediction for the opposite case is untested, and the argument of Section 3.7 is heuristic. The alignment uses the training outcome and is noisy at n = 50.

**Evaluation target.** We score retrieval through the prediction it supports and through neighbour precision. Whether better-scoring neighbours also improve a downstream system, for example a language model that reads them, is not measured.

**Repetitions.** Results use {{reps_min}} repetitions per cell. The runner is resumable and further blocks can be added; the manuscript is rebuilt from the outputs.
