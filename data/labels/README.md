# Labels CSV schema

Required columns:

- `flow_id` — must match extracted feature `flow_id` values
- `label` — `benign`/`suspicious` or `0`/`1`

When a valid labels file is configured (`labels.path` in settings or `ETTD_LABELS_PATH`), the system trains a Random Forest. Otherwise Isolation Forest is used.

Do not commit real production labels containing sensitive identifiers unless approved.
