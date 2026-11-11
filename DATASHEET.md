# Datasheet for the learner-similarity benchmark views

This benchmark uses **public, already anonymised datasets only**. No data from the authors'
institution, no transcripts and no personally identifying attribute is used.

## 1. Sources

All files were retrieved on 2026-11-11. `data/raw/MANIFEST.json` records the URL, size and
SHA-256 of every downloaded file. Total download: about 1.6 GB.

| Source | What it is | Licence (as stated by the provider) | Where it was downloaded from |
|---|---|---|---|
| OULAD | Open University Learning Analytics Dataset: 7 modules, 22 presentations, VLE clicks | CC BY 4.0 | UCI Machine Learning Repository, dataset 349 (the link on the Open University page returned HTTP 404 on the retrieval date) |
| PTADisc | Programming Teaching Assistant logs, NeurIPS 2023 Datasets and Benchmarks | CC BY-NC-SA 4.0 (LICENSE file in the repository) | `huggingface.co/datasets/whning0513/PTADisc`, folders `cross_course_datasets` and `baseline_dataset_for_CD` only |
| XES3G5M | Grade-3 mathematics, NeurIPS 2023 Datasets and Benchmarks | MIT (repository LICENSE) | Google Drive archive linked from `github.com/ai4ed/XES3G5M` |
| EdNet-KT1 sample | 20,000-learner sample of EdNet-KT1 prepared by the DKT2 authors | EdNet: CC BY-NC 4.0; DKT2 code: MIT | Google Drive archive linked from `github.com/zyy-2001/DKT2` |
| ASSISTments 2017 | Data-mining competition release, pre-processed by the DKT2 authors | not stated in the files; see the ASSISTments site | same DKT2 archive |
| ASSISTments 2009 | Skill-builder data 2009-2010, "corrected, collapsed" release | not stated in the file; see the ASSISTments site | Google Drive file linked from the ASSISTments data site |
| ASSISTments 2015 | 2015 skill-builder data (100 skills) | not stated in the file; see the ASSISTments site | Google Drive file linked from the ASSISTments data site |
| Slepemapy | Adaptive geography practice (Papoušek, Pelánek & Stanislav, 2016) | Open Database License; contents under the Database Contents License (README in the archive) | `fi.muni.cz/adaptivelearning`, file `2015-05-21.zip` |

The three ASSISTments files carry no licence text. Check the terms on the ASSISTments data
site before redistributing them. This repository redistributes **no raw data**: it ships
the download script, the build script and the checksums.

Non-commercial licences (PTADisc, EdNet) restrict the use of these data to research.
PTADisc is share-alike: derived data must be released under the same licence.

## 2. Columns that are never read

| Source | Columns deliberately not loaded |
|---|---|
| OULAD | `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`; assessment scores |
| Slepemapy | `ip_country`, `ip_id`, `response_time`, `language`, `options` |
| ASSISTments 2009 | teacher, school and class identifiers, answer text |

The OULAD feature set is restricted to registration timing, number of previous attempts,
studied credits, VLE activity in the first 28 days and the number of assessments handed in
by day 28. No demographic attribute is used in any analysis, including robustness checks.

## 3. What a "view" is

A view is a learner-level table: one row per learner, `d` ability features, one outcome,
and a fixed partition of the learners. The construction rule is in
`docs/FEATURE_RULE.md` and implemented in `src/lsb/build_views.py`. The list of views with
their sizes is `data/processed/views_index.csv` (and Table 1 of the manuscript).

Learners are split once per source with seed 2026:

- **reference pool** (20 %, at most 3,000 learners): used only to define the features;
- **test set** (at most 1,000 learners): never used to fit anything;
- **candidate pool** (the rest): training samples of size n are drawn from it.

## 4. Known limitations of the data

- PTADisc logs carry no timestamps, so the PTADisc views use cross-course and
  unseen-skill outcomes instead of a temporal split.
- The EdNet and ASSISTments 2017 files are third-party pre-processed releases. In EdNet
  each distinct combination of tags is one knowledge component.
- The outcome of the "future performance" views is a raw success rate on later items. It
  is not adjusted for item difficulty.
- The five PTADisc cross-course views share the same 29,454 learners, and the six OULAD
  views come from one institution. Views are therefore grouped into eight independent
  sources in every statistical analysis that needs independence.
- None of the sources is a university transcript, and none is from Vietnam.
