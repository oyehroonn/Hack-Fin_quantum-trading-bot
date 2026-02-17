"""LLM integration: governor, sentinel, report writer.

The LLM is NOT a predictor — it is a governor/orchestrator that:
  1. Selects strategies based on market context
  2. Triages anomalies
  3. Generates human-readable reports
  4. Adjusts risk parameters

All LLM outputs go through strict JSON schema validation.
"""
