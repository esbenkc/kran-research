# Deviations — decoy-usefulness

Changes after PREREG v1 (a35a72c). Git time is the record. The frozen rule is still reported next to any changed one.

## D1. Descriptive split by logs-bucket seeds (Vera #10191; written during the smoke, before any counted run)

Not a rule change: a reported, descriptive row only. In 6 of the counted seeds 0-19 (1, 2, 4, 6, 15, 19) the
workspace bucket is named `bucket-logs-*`, and the task also sets a log level. A `log export reads` key may then
look half-useful. That would lower the usefulness effect, so the bias is against YES. Vera reports D for the 6
logs-bucket seeds and the 14 other seeds separately. The status uses all 20 seeds, as frozen.
