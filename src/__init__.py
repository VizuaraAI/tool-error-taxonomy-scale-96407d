"""Tool-error taxonomy across scale: measurement code for the study.

Layout:
  taxonomy.py    the seven error types and the cascade order
  parsing.py     turn a raw completion into structured tool calls
  attribution.py the cascade itself (the measuring instrument)
  data.py        benchmark loaders, normalised to one instance schema
  models.py      HuggingFace generation backend for the scale ladder
  remedies.py    the seven type-matched remedies
  evaluate.py    entry point: run a model over a benchmark, write results
  analyze.py     entry point: error mass, elasticity, figures
  train_lora.py  optional: small LoRA fine-tune for the week 7 extension
"""

__version__ = "0.1.0"
