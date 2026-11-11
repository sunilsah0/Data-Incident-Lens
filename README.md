# DataIncidentLens

![tests](https://github.com/sunilsah0/Data-Incident-Lens/actions/workflows/tests.yml/badge.svg)

**An open-source control tower for data and AI reliability.** When data breaks, it groups the
alerts, ranks the likely root cause, shows the blast radius, plans a cheaper recovery, and
suggests a test to prevent a repeat. Every claim below is measured in a fault-injection lab.

**Live demo:** <your Streamlit link>  |  **Demo video:** <your link>

## Why
Commercial tools such as Monte Carlo, Bigeye and Datadog are paid, and their AI root-cause
agents are black boxes with marketing-level accuracy claims. To my knowledge there is no
public, reproducible benchmark. IncidentLens is free, transparent (every score is shown)
and benchmarked.

## What it covers (10 dashboard modules)
Overview, Data Quality, Pipeline Monitoring, Real-Time Analytics, Cost Intelligence,
ML Monitoring, AI Reliability, Governance & Security, Incident Management, AI Engineering Assistant.

## Architecture
<diagram: sources -> dbt/DuckDB -> collectors -> metadata store -> engine -> Streamlit>

## Results
All results come from a 15-table lab built from NYC taxi data with 18 injected fault types.
Scripts are in `experiments/`; raw rows are in `experiments/results/`.

| Experiment | Result |
|---|---|
| Fault detection (108 runs) | 100% of 18 fault types detected |
| Alert grouping | 8.5 alerts per fault become 1 incident |
| Root cause, single fault | 100% top-1, **but loudest-alert and most-upstream baselines also score 100%**, so this lab cannot separate the methods |
| Root cause, two faults at once (40 pairs) | Recall 0.71 vs 0.50 for both baselines; both causes found in 42% vs 0% |
| Dependency analysis | Column-level F1 0.975 vs 0.924 for table-level (recall 1.0 for both) |
| Change prediction (26 changes) | F1 0.91 vs 0.76 ("any change is risky") and 0.53 (schema diff); 0 breaking changes missed |
| Business impact | <paste results of experiments.impact> |
| Recovery planning (18 scenarios) | 26% of full-replay cost vs 39% for partition backfill, 100% correct |
| AI trust (144 answers) | Grounding-only check recall 0.48 (misses all stale and unauthorised answers); catalogue-aware check 1.00 |
| Cost optimisation (40 setups) | 17% of original cost with 97.0% of reads meeting freshness limits; blind cut costs the same but meets 71.9% |

## Limitations (read these)
- Faults, access logs, streaming and cloud prices are **simulated**; business values are synthetic.
- Nested faults (one fault downstream of another, 23 of 40 pairs) are often explained away by the upstream root cause.
- Column-level lineage treats columns used in WHERE clauses as affecting every output (e.g. `unit_distance` precision 0.38).
- Change prediction compares normalised SQL, so equivalent rewrites (`reorder_predicates`, `reorder_in_list`) cause false alarms; a LEFT JOIN change was flagged although no row differed in the sample.
- The AI trust checker and the faults share the same metadata fields, so its perfect score is partly by construction. The grounding-only blind spot is the real finding.
- Cost savings are measured against an "everything hourly" schedule; compliance dropped from 100% to 97% because of reads the usage log never saw.
- Recovery's validate-and-escalate step was never triggered in the 18 scenarios.

## Run it
```
pip install -r requirements.txt
python collectors/load_raw.py
dagster dev -f orchestration/definitions.py
streamlit run dashboard/app.py
python -m experiments.run_experiments
```

## Future work
Residual-based root cause for nested faults, filter-aware column lineage, a real document
versioning feed for the AI trust check, real warehouse billing data.
