# stage 01 - query classification

the LLM pipeline that coded learner queries into the 16 task codes

- `pipeline.py` - classification pipeline
- `input/` - synthetic sample conversation logs
- `data/sample_codes.csv`, `data/sample_solves.csv` - synthetic classifier output + solves

## Usage

```bash
# no API key: assemble the stage-02 modeling input from the synethic data
python3 pipeline.py --assemble-only --input-dir input \
    --output-csv data/sample_codes.csv --inputfeatures-csv inputfeatures_demo.csv --solves-csv data/sample_solves.csv

# live classification, --dry-run prints the first prompt
OPENROUTER_API_KEY=sk-or-...  python3 pipeline.py --provider openrouter --input-dir input --output-csv out.csv
```
