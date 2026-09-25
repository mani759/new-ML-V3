# MPLADS Anomaly Detection - ML Handoff (v3, real eSAKSHI data)

SIH26102 · ML side: Pallavi · Snapshot: 24 September 2026 · Model version 3.0.0

This package replaces every earlier ML file (v1 synthetic model and the v2 service fixes).
The model now scores **136,469 real MPLADS works** from MoSPI's public eSAKSHI dashboard
(18th Lok Sabha + Rajya Sabha). Read with the *eSAKSHI Data Understanding* doc, which explains
the data, why Telangana and Ladakh needed special handling, and the agreed model design.

If you work with Claude: give it this file, `schema.sql`, `app.py` and the data-understanding doc.

---

## 1. What changed from the prototype

| Before (prototype) | Now (v3) |
| --- | --- |
| 3,364 synthetic projects (`mplads_projects.csv`) | 136,469 real works + 113,129 real payments |
| Rules on invented fields (tender, expected completion) | Rules on real fields and official guidelines (45-day sanction, 1-year completion) |
| One `projects` table | `works`, `payments`, `work_scores`, `mp_alerts`, `review_actions` (see `schema.sql`) |
| `/score` + `/duplicates` over HTTP for every project | Bulk scoring runs as a job (`pipeline.py`); HTTP only for the live "Analyze a work" form |
| No real prediction | Delay-risk model trained on real outcomes |

**Delete from the app:** the synthetic dataset, `datasetGenerator.ts`, the old duplicate fallback in
`mlService.ts`, every `ground_truth_*` field, and all "TF-IDF" wording.

---

## 2. Package contents

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI service: `/`, `/version`, `/score`, `/score/batch` |
| `risk_engine.py` | All features, rules, reasons, scoring (shared by batch and live) |
| `pipeline.py` | Refresh job: `fetch` from eSAKSHI -> `build` clean tables -> `score` |
| `train_and_validate.py` | Rebuilds benchmarks + models and the validation report (run rarely) |
| `live_acceptance_tests.py` | 11 known-answer tests; must all pass after every deploy |
| `artifacts/bench.pkl`, `artifacts/models.pkl` | Trained benchmarks, Isolation Forest, delay model |
| `data/works.csv`, `data/payments.csv` | Clean real data (load into DB) |
| `data/alloc_LS.csv`, `data/alloc_RS.csv` | MP allocations (eSAKSHI export) |
| `output/scored_works.csv`, `output/mp_alerts.csv` | Current scores (load into DB) |
| `output/validation_report.json` | Accuracy numbers to show on the dashboard and in the pitch |
| `build_dataset.py` | How the clean data was built from the raw browser downloads (provenance) |
| `schema.sql` | Database tables |

---

## 3. Deploy the ML service

```bash
pip install -r requirements.txt           # versions are pinned; keep them
export ML_SERVICE_KEY=<random secret>      # same value in the Node backend .env
uvicorn app:app --host 0.0.0.0 --port 8000
python live_acceptance_tests.py https://<ml-url> --key <secret>   # all 11 must PASS
```

- The service loads `artifacts/` at start-up (about 10 MB). Use an always-on instance for the demo
  period; free tiers that sleep will time out on the first request.
- Every call from Node sends header `X-Service-Key`.

---

## 4. API contract

### POST /score - live "Analyze a work" form (read-only, never writes to the DB)

Request (required fields first; the rest describe progress if known):

```json
{
  "house": "LS", "state": "Bihar", "ida": "PATNA(DISTRICT PLANNING OFFICER PATNA_IDA)",
  "mp_name": "Test MP",
  "activity_type": "Construction of roads, link roads, pathways or any other road with or without drainage system",
  "work_description": "Construction of PCC road from main road to Test Tola, Ward 5",
  "recommended_amount": 990000, "recommendation_date": "2025-12-01",
  "sanction_date": "2026-01-10", "in_sanctioned_list": true, "stage": "Vendor Identification",
  "total_paid": 940500, "payment_count": 1, "vendor_count": 1, "last_payment_date": "2026-01-20"
}
```

- `ida` and `activity_type` must be exact eSAKSHI values: offer them as dropdowns built from the
  `works` table (777 districts, 115 activity types).
- `as_of` (optional) = scoring date. Omit for live use (today).

Response:

```json
{
  "work_id": null, "risk_score": 75, "severity": "high",
  "flags": ["paid_but_stalled"],
  "reasons": ["95% of Rs 9.9 L paid, but the work is still at 'Vendor Identification' and nothing has been paid for 247 days."],
  "features": {"amount": 990000.0, "peer_median": 992160.0, "cost_ratio": 1.0, "kind": "construction",
               "district_sanction_p90_days": 69, "days_since_sanction": 257.0, "paid_frac": 0.95},
  "ml_anomaly_score": 0.4149, "delay_risk": 0.01, "as_of": "2026-09-24", "model_version": "3.0.0"
}
```

- `reasons` has one plain-language sentence per flag, in the same order. **Show them as the main
  explanation.** Gemini may rephrase them, but the original sentences must stay visible.
- `features` = the values the decision used; show them as a small "why" table.
- `delay_risk` is only filled for open works under 365 days old; otherwise `null`.
- A live duplicate check against all stored works is included automatically.

### GET /version
Model version, data snapshot date and the validation summary (use it for the "Model validation" panel).

### Flags and severities

| Flag | Severity | Meaning |
| --- | --- | --- |
| `overdue_stalled` | high | Open over 365 days after sanction; no payment ever or none in 180 days |
| `overdue` | medium | Open over 365 days after sanction; payments still moving |
| `sanction_pending` | medium / high over 1 year | Waiting over 45 days and beyond the district's 90th percentile |
| `unusually_expensive` | medium | Over 4x the median of similar works and a statistical outlier |
| `bundled_work` | low | Covers many locations; unit cost not verifiable |
| `vague_description` | low | Description gives no detail, above the peer median |
| `completed_far_below_sanction` | medium | Actual cost under 50% of sanction |
| `duplicate_paid` | high | Same MP, same location-specific description, amount within 10%, all copies paid |
| `possible_duplicate` | medium | As above, not all copies paid |
| `same_work_other_mp` | medium | Same specific work in same district recommended by another MP |
| `paid_but_stalled` | high | 90%+ paid, still at an early stage, no payment in 180 days |
| `construction_same_day` | high | Construction completed on the sanction date |
| `construction_within_7_days` | medium | Construction completed 1-7 days after sanction |
| `no_recommendation_record` | medium | Sanctioned but missing from the recommended list |
| `completed_without_payment` | medium | Completed with no payment recorded |
| `repeated_payment_entries` | low | Same payment recorded twice (likely instalments) |
| `statistical_anomaly` | +30 | Isolation Forest; only on works where no rule fired |

Score = combined probability of the flag weights (high 75, medium 40, low 12, ML 30).
Severity: high >= 70, medium >= 40, low >= 12, else none. So one High flag = high, and two Medium
flags (e.g. expensive and overdue) = medium 64, three = high.

MP-level alerts (`mp_alerts`): `vendor_concentration` (largest vendor over 60% of an MP's payments,
10+ paid works; 61 MPs) and `low_fund_utilisation` (recommended under 25% of allocation after a full
year; 40 MPs).

---

## 5. Data refresh (daily job)

```bash
python pipeline.py fetch --discover   # FIRST run only: finds every eSAKSHI state code (e.g. Telangana = 129)
python pipeline.py all                # daily: fetch + build + score
```

Then replace the `works`, `payments`, `work_scores` and `mp_alerts` tables with the new files
(load into staging tables and swap, so the dashboard never sees half-loaded data).

- `fetch` calls the dashboard's own request state by state, with pauses. It is **not a documented
  API** and could not be tested from our build environment. Run it once by hand and check the totals
  against the dashboard tiles (Section 7) before scheduling it.
- If `fetch` is blocked, the manual browser method in the data-understanding doc still works; save
  the files into `raw/` and run `pipeline.py build` and `score`.
- `pipeline.py score` uses the trained artifacts. Retrain (`train_and_validate.py`) only on purpose,
  e.g. monthly, then re-run the acceptance tests.

---

## 6. Dashboard requirements (functional, not visual)

1. **Four stakeholder views = four filters on the same data** (PS requirement):
   Ministry = all; State Nodal Authority = `state`; District Authority = `ida`;
   MP = `house` + `mp_name`. Every count, chart and alert comes from the same filtered set.
2. **One definition of "flagged":** `severity IN ('medium','high')`. Use it in every endpoint.
3. **Sort by `risk_score` descending** with district filters. 16,289 works are High, mostly
   stalled works past the 1-year guideline; officers need their own top cases first.
4. **Show `reasons` and `features`** on every flagged work and in the live form result.
5. **Show `as_of` and `model_version`** next to headline numbers ("Scored as of 24 Sep 2026 - v3.0.0").
6. **Delay-risk list** for District Authorities: open works with `delay_risk > 0.7` (13,629 today),
   labelled "predicted risk", not an accusation.
7. **MP-level panel** from `mp_alerts`.
8. **Model validation panel** from `/version` (Section 7), labelled
   "Validated by planting known anomalies into real works".
9. **Historical context (optional)**: data.gov.in state figures 2016-20 next to eSAKSHI state
   figures, same formula. Not part of the risk score.
10. **Reviewer workflow** writes to `review_actions` (under_review, verified, escalated,
    clarification_requested, false_positive, resolved). Workflow status is separate from severity.

---

## 7. Validation (from output/validation_report.json)

Real data has no fraud labels, so each check was tested by planting 500 known anomalies into copies
of real, clean works and scoring them with everything else.

| Check | Caught by the intended check |
| --- | --- |
| Overdue and stalled | 100% |
| Paid but stalled | 100% |
| Construction completed same day | 100% |
| No recommendation record | 100% |
| Sanction pending over 1 year | 94.6% (100% flagged overall) |
| Duplicate, all paid | 93.2% (misses fall into repeated standard-item batches, by design) |
| Unusually expensive (6x peers) | 64.8% (activity types mix small and large works; no quantity field) |
| Multi-factor anomaly, no single rule broken | 82.2% by Isolation Forest (100% flagged overall) |

Delay prediction, trained on works sanctioned before April 2025 and tested on April-September 2025
(27,778 works it never saw): ROC-AUC 0.758 (district-history baseline 0.67); at 50%: precision 0.68,
recall 0.56, F1 0.62.

Totals check against the portal tiles: all within 0.1% (portal is live; tiles and downloads were read
at different moments). Ladakh 55 works / Rs 5.45 Cr and Telangana 5,374 LS works match their tiles
exactly.

---

## 8. Still open from Change Request Report v2

| Item | Action |
| --- | --- |
| N1 leaked secrets | Rotate Gemini key, Supabase service-role key, DB password; never ship `.env` |
| N2 ground truth leak | Gone with the synthetic data; make sure no `ground_truth_*` field survives |
| N7 cold start | Always-on ML instance; run acceptance tests before every demo |
| N8 live form writes | `/score` is read-only; keep the Node route read-only too |
| N10 / N11 | Workflow status separate from severity; one "flagged" definition (Section 6) |
| N13 service auth | `ML_SERVICE_KEY` + `X-Service-Key` header |
| W1-W3 wording | Remove "TF-IDF", "real-time ingestion", "predict delays" -> "delay risk", Random Forest etc. in deck |
| AI Studio | Deploy on regular hosting with the team's own URL |

---

## 9. Pitch-safe wording

- "Real works and payments from MoSPI's public eSAKSHI dashboard" (not "eSAKSHI API").
- "Historical state-level context from data.gov.in via its open-data API."
- "Rules grounded in MPLADS guidelines and CAG-reported patterns, plus an Isolation Forest for
  unusual combinations, plus a delay-risk model trained on real outcomes."
- "Validated by planting known anomalies into real works" (Section 7 numbers).
- Say the limits: no cost-estimate, tender or GPS data in the public dashboard; cost checks are
  peer-based; GPS verification of repeated items is future scope.
