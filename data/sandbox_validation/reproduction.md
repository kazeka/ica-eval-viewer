# Reproduction of ica_sandbox.evalval

| scorer                     | check                                    |   evalval |   ica_eval | match   |
|:---------------------------|:-----------------------------------------|----------:|-----------:|:--------|
| Lookalike (current method) | T2 conversion ΔAUC                       |    0.0784 |     0.0784 | True    |
| Lookalike (current method) | T3 accuracy                              |    0.6040 |     0.6040 | True    |
| Lookalike (current method) | T4 SME agreement                         |    0.6446 |     0.6446 | True    |
| Lookalike (current method) | T5 remove the answer                     |    0.9400 |     0.9400 | True    |
| Lookalike (current method) | T5 answer a different objection          |    0.8291 |     0.8291 | True    |
| Lookalike (current method) | T5 add invented offer (known to rules)   |    0.3350 |     0.3350 | True    |
| Lookalike (current method) | T5 add invented offer (unknown to rules) |    0.3750 |     0.3750 | True    |
| Lookalike (current method) | T5 add pushy line                        |    0.4950 |     0.4950 | True    |
| Lookalike (current method) | T5 make it too long                      |    0.9400 |     0.9400 | True    |
| Lookalike (current method) | T6 paraphrase (same meaning)             |    0.1030 |     0.1046 | False   |
| Lookalike (current method) | T6 add speech fillers (same meaning)     |    0.0338 |     0.0338 | True    |
| Outcome model: next turn   | T2 conversion ΔAUC                       |    0.1189 |     0.1189 | True    |
| Outcome model: next turn   | T3 accuracy                              |    0.6613 |     0.6613 | True    |
| Outcome model: next turn   | T4 SME agreement                         |    0.6675 |     0.6675 | True    |
| Outcome model: next turn   | T5 remove the answer                     |    1.0000 |     1.0000 | True    |
| Outcome model: next turn   | T5 answer a different objection          |    0.9500 |     0.9500 | True    |
| Outcome model: next turn   | T5 add invented offer (known to rules)   |    0.0150 |     0.0150 | True    |
| Outcome model: next turn   | T5 add invented offer (unknown to rules) |    0.3600 |     0.3600 | True    |
| Outcome model: next turn   | T5 add pushy line                        |    0.7750 |     0.7750 | True    |
| Outcome model: next turn   | T5 make it too long                      |    1.0000 |     1.0000 | True    |
| Outcome model: next turn   | T6 paraphrase (same meaning)             |    0.4835 |     0.4909 | False   |
| Outcome model: next turn   | T6 add speech fillers (same meaning)     |    0.2847 |     0.2847 | True    |
| Outcome model: sale        | T2 conversion ΔAUC                       |    0.1151 |     0.1151 | True    |
| Outcome model: sale        | T3 accuracy                              |    0.6784 |     0.6784 | True    |
| Outcome model: sale        | T4 SME agreement                         |    0.6525 |     0.6525 | True    |
| Outcome model: sale        | T5 remove the answer                     |    0.9600 |     0.9600 | True    |
| Outcome model: sale        | T5 answer a different objection          |    0.9000 |     0.9000 | True    |
| Outcome model: sale        | T5 add invented offer (known to rules)   |    0.5200 |     0.5200 | True    |
| Outcome model: sale        | T5 add invented offer (unknown to rules) |    0.9400 |     0.9400 | True    |
| Outcome model: sale        | T5 add pushy line                        |    0.8800 |     0.8800 | True    |
| Outcome model: sale        | T5 make it too long                      |    0.8650 |     0.8650 | True    |
| Outcome model: sale        | T6 paraphrase (same meaning)             |    0.5551 |     0.5636 | False   |
| Outcome model: sale        | T6 add speech fillers (same meaning)     |    0.3670 |     0.3670 | True    |
| Rule checks                | T4 SME agreement                         |    0.5950 |     0.5950 | True    |
| Rule checks                | T5 remove the answer                     |    0.0000 |     0.0000 | True    |
| Rule checks                | T5 answer a different objection          |    0.0000 |     0.0000 | True    |
| Rule checks                | T5 add invented offer (known to rules)   |    1.0000 |     1.0000 | True    |
| Rule checks                | T5 add invented offer (unknown to rules) |    0.0000 |     0.0000 | True    |
| Rule checks                | T5 add pushy line                        |    0.0000 |     0.0000 | True    |
| Rule checks                | T5 make it too long                      |    1.0000 |     1.0000 | True    |
| Rule checks                | T6 paraphrase (same meaning)             |    0.0000 |     0.0000 | True    |
| Rule checks                | T6 add speech fillers (same meaning)     |    0.0000 |     0.0000 | True    |
| Lookalike (current method) | T1 mean rank correlation                 |    0.2986 |     0.2986 | True    |
