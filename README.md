# Prescription Audit Eval

This project evaluates prescription-audit models from three sources:

- local OpenAI-compatible servers
- remote OpenAI-compatible servers
- third-party APIs

## Layout

- `run_eval.py`: entry point
- `configs/example_config.json`: example configuration
- `src/prescription_audit/config.py`: config loading
- `src/prescription_audit/models.py`: model client layer
- `src/prescription_audit/parsing.py`: output parsing
- `src/prescription_audit/metrics.py`: metric computation
- `src/prescription_audit/pipeline.py`: end-to-end evaluation

## Run

```bash
python run_eval.py --config configs/example_config.json
```

## Notes

- Local models stay registered by `path`, then get mapped to runtime `base_url`.
- CSV files are written with `utf-8-sig`.
- Excel reports are written with `openpyxl`.
