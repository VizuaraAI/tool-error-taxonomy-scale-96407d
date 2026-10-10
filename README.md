# tool-error-taxonomy-scale

**Which error comes first? Mapping the shift in tool-use error types across
0.5B, 1B and 3B language models.**

Starter repository for Balakrishnan S, Vizuara SLM Bootcamp.

---

## The one-sentence version

Everybody reports a single tool-calling accuracy number, but a model can fail
in seven distinct ways, each with a different fix. This project measures how
the *mix* of those seven failures changes between 0.5B, 1.5B and 3B, and then
checks that the matching fix actually helps where the map says it should.

The seven error types, following the ToolScan taxonomy
([Kokane et al., 2024](https://arxiv.org/abs/2411.13547)):

| code | error type         | family   | the fix that targets it      |
| ---- | ------------------ | -------- | ---------------------------- |
| E7   | invalid formatting | surface  | grammar-constrained decoding |
| E6   | wrong function name| surface  | tool-document retrieval      |
| E3   | wrong arg names    | surface  | schema name canonicalisation |
| E4   | wrong arg types    | bridge   | type coercion repair         |
| E2   | wrong arg values   | semantic | value-grounding exemplars    |
| E1   | insufficient calls | semantic | call-set planning prompt     |
| E5   | repeated calls     | semantic | duplicate-call state tracker |

The hypothesis: error mass migrates from the surface family at 0.5B to the
semantic family at 3B, so the best remedy migrates with it.

---

## Setup

```bash
git clone <your-fork> tool-error-taxonomy-scale
cd tool-error-taxonomy-scale
./scripts/setup.sh
source .venv/bin/activate
```

`setup.sh` installs dependencies and then runs the cascade self-test and the
unit tests, so you know the measuring instrument works before you download any
model weights.

You can read and test the science with **no dependencies at all**:
`src/taxonomy.py`, `src/parsing.py` and `src/attribution.py` are pure standard
library.

```bash
python3 -m src.attribution --selftest
```

---

## The first three things to run

**1. Convince yourself the instrument is right.** Nine hand-written cases, one
per error type plus the tricky prose-versus-malformed distinction.

```bash
python -m src.attribution --selftest
pytest -q
```

Open `src/attribution.py` and read the `attribute` function top to bottom. It
is about 120 lines and it is the whole scientific contribution of the project.
If you disagree with a decision in the cascade, change it now, before you have
results that depend on it.

**2. Get one real number.** The smallest model, the smallest benchmark slice.

```bash
LIMIT=50 ./scripts/run_eval.sh 0.5B bfcl
```

This downloads Qwen2.5-0.5B-Instruct (about 1 GB) and writes
`runs/0.5B_bfcl_none_seed0.jsonl` plus a summary. Look at the summary's
`error_mass_share`: that is your first data point, and it is already a result
nobody has published.

**3. Get the shape of the whole thing.** Three rungs, baselines only.

```bash
LIMIT=200 ./scripts/run_sweep.sh
python -m src.analyze --runs runs --out analysis
```

Then open `analysis/error_mass.csv`. If the surface share falls and the
semantic share rises as you go down the ladder, the project's central claim is
alive and you should go read `analysis/elasticity.csv` next.

---

## Layout

```
src/
  taxonomy.py      the seven types, the cascade order, the two families
  parsing.py       raw completion -> list[ToolCall], deliberately generous
  attribution.py   the cascade; one primary label plus every co-occurring one
  data.py          benchmark loaders, all normalised to one Instance shape
  models.py        the scale ladder behind one thin HuggingFace wrapper
  remedies.py      the seven type-matched remedies (prompt-side and post-hoc)
  evaluate.py      entry point: generate, repair, attribute, write JSONL
  analyze.py       entry point: error mass, elasticity, remedy gain, figure
  train_lora.py    optional week 7 extension, a small LoRA fine-tune
configs/default.yaml   every knob the paper has to report
scripts/               setup, single run, full sweep
experiments/           your lab notebook; rows map to the paper's tables
tests/                 regression tests for the instrument
```

No data files are committed. Loaders pull from the Hugging Face Hub.

---

## Benchmarks

| benchmark | source | size | why it is here |
| --------- | ------ | ---- | -------------- |
| BFCL v3 | [`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | ~12 MB, Apache 2.0 | AST categories give five of the seven types directly |
| NESTFUL | [`ibm-research/nestful`](https://huggingface.co/datasets/ibm-research/nestful) | 1,861 rows, Apache 2.0 | nested call sequences; the only place E1 and E5 can occur |
| API-Bank | [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | 314 dialogues | turn-level timing failures |

Start with BFCL AST categories: they need no third-party API keys. Avoid
ToolBench as a primary benchmark; its evaluation needs a gated server key, a
RapidAPI key and an OpenAI key, which is three ways for your experiment to
break for reasons that have nothing to do with your research question.

---

## Hardware

The whole study fits on one GPU with 16 GB, and the 0.5B and 1.5B rungs fit on
a laptop if you are patient. A 200-instance run at 0.5B takes a few minutes; the
full grid (3 models x 8 conditions x 3 seeds) is an overnight job. Run
baselines first and look at the error profile before you spend a night on the
grid.

---

## Where this goes

The paper draft lives alongside this repo (`paper/main.tex`). Its Table 1 is
`analysis/error_mass.csv`, its Table 2 is `analysis/remedy_gain.csv`, and its
Figure 2 is regenerated by `src/analyze.py`. Replace the `XX.X` placeholders as
the numbers arrive; do not wait until the end to start writing.
