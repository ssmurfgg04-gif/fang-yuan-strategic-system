# Fang Yuan Deterministic Benchmark — merged report

- shards merged: **10** (all present)
- layers: **canon, counterfactual, dynamic** (deterministic CI layers only)
- items: **118** × agents: **fang_yuan_policy_v1, safe_generic, reckless_theatrical**

## Global F score (0.20O + 0.15A + 0.15R + 0.15I + 0.15P + 0.10M + 0.10C)

| agent | items | mean item score | F |
|---|---:|---:|---:|
| fang_yuan_policy_v1 | 118 | 0.8925 | **0.802** |
| safe_generic | 118 | 0.4740 | **0.317** |
| reckless_theatrical | 118 | 0.4098 | **0.194** |

## Per-layer breakdown (mean item score)

| layer | items | fang_yuan_policy_v1 | safe_generic | reckless_theatrical |
|---|---:|---:|---:|---:|
| canon | 40 | 0.8021 | 0.0250 | 0.0000 |
| counterfactual | 33 | 1.0000 | 0.5455 | 0.5455 |
| dynamic | 45 | 0.8941 | 0.8206 | 0.6745 |

