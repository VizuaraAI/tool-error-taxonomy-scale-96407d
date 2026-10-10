# Experiment log

Copy this file to `experiments/log.md` and append one row per run. Never edit
an old row: if a run was wrong, add a new row and say so in Notes.

The columns below map directly onto the paper:

| this file                  | paper                         |
| -------------------------- | ----------------------------- |
| Table A rows               | Table 1 (error mass by scale) |
| Table B rows               | Table 2 (remedy x scale gain) |
| Table A `alpha`            | the `alpha` column of Table 1 |
| Table C                    | the instrument-validation paragraph in Section 5 |

Every number here comes from `runs/<stem>.summary.json`, so you should never be
typing a number you computed in your head.

---

## Table A: baseline error mass by scale

One row per (model, benchmark, seed) with no remedy applied. Rates are the
percentage of *all* instances; shares are the percentage of that model's total
error mass. Shares across the seven types sum to 100.

| date | model | benchmark | n | seed | call acc | total err | E7 rate/share | E6 rate/share | E3 rate/share | E4 rate/share | E2 rate/share | E1 rate/share | E5 rate/share | commit | notes |
| ---- | ----- | --------- | - | ---- | -------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------ | ----- |
|      | 0.5B  | bfcl      |   | 0    |          |           |     /         |     /         |     /         |     /         |     /         |     /         |     /         |        |       |
|      | 1.5B  | bfcl      |   | 0    |          |           |     /         |     /         |     /         |     /         |     /         |     /         |     /         |        |       |
|      | 3B    | bfcl      |   | 0    |          |           |     /         |     /         |     /         |     /         |     /         |     /         |     /         |        |       |

### Derived: scale elasticity

Fill from `analysis/elasticity.csv` once you have all three rungs. Strongly
negative means scale alone dissolves the type; near zero means it will not.

| error type | family   | alpha | reading |
| ---------- | -------- | ----- | ------- |
| E7         | surface  |       |         |
| E6         | surface  |       |         |
| E3         | surface  |       |         |
| E4         | bridge   |       |         |
| E2         | semantic |       |         |
| E1         | semantic |       |         |
| E5         | semantic |       |         |

---

## Table B: remedy x scale gain

Gain is absolute change in call accuracy against the matching no-remedy row in
Table A. Same benchmark, same seed, same limit, or the number is meaningless.

| date | remedy | target type | 0.5B gain | 1.5B gain | 3B gain | tracks its target's share? | commit | notes |
| ---- | ------ | ----------- | --------- | --------- | ------- | -------------------------- | ------ | ----- |
|      | grammar_constrained_decoding | E7 |  |  |  |  |  |  |
|      | tool_doc_retrieval           | E6 |  |  |  |  |  |  |
|      | schema_name_canonicalisation | E3 |  |  |  |  |  |  |
|      | type_coercion_repair         | E4 |  |  |  |  |  |  |
|      | value_grounding_fewshot      | E2 |  |  |  |  |  |  |
|      | call_set_planning_prompt     | E1 |  |  |  |  |  |  |
|      | duplicate_call_state_tracker | E5 |  |  |  |  |  |  |

---

## Table C: instrument validation

Hand-label a held-out slice of 100 instances, then compare against the
cascade. This number belongs in the main paper: if the instrument disagrees
with you mainly on E2 versus E3, the central comparison is undermined.

| date | n hand-labelled | overall agreement | worst-confused pair | notes |
| ---- | --------------- | ----------------- | ------------------- | ----- |
|      |                 |                   |                     |       |

---

## Table D: cross-family replication

The question: does a 1.7B model trained on a different data mixture look like
the 1.5B rung (scale effect) or like the 0.5B rung (post-training effect)?

| date | model | params (B) | benchmark | surface share | semantic share | closest ladder rung | notes |
| ---- | ----- | ---------- | --------- | ------------- | -------------- | ------------------- | ----- |
|      | llama-1B  | 1.24 |  |  |  |  |  |
|      | llama-3B  | 3.21 |  |  |  |  |  |
|      | smol-360M | 0.36 |  |  |  |  |  |
|      | smol-1.7B | 1.71 |  |  |  |  |  |

---

## Running notes

Date-stamped prose. This is where the paper's Discussion section comes from, so
write the surprises down the day they happen, not the week you draft.

### YYYY-MM-DD
- What I ran:
- What I expected:
- What actually happened:
- What I will do next:
