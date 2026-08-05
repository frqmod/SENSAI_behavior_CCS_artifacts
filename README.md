# Do Hackers Dream of Electric Teachers?

Public companion code for the ACM CCS 2026 work *Do Hackers Dream of Electric
Teachers?: A Large-Scale, In-Situ Measurement of Cybersecurity Student Behaviors
and Educational Performance with AI Tutors*.

- `run_all.sh` - demo both stages on the synthetic data (no API key needed)
- `01_coding/` - the LLM query-coding pipeline
- `02_mhmm/` - the conversation-style MHMM fit
- `Dockerfile` - pinned R + Python environment

## Requirements

- **R** >= 4.0 + `Rscript -e 'install.packages(c("seqHMM","dplyr","tidyr"))'`
- **Python** >= 3.9 + `pip3 install openai pydantic`

## Usage

```bash
bash run_all.sh    # stage 01 assemble demo -> stage 02 MHMM fit

# or containerized for exact reproducibility
docker build -t sensai-public .
docker run --rm -v "$PWD":/artifact sensai-public bash -c "bash run_all.sh 2>&1 | tee run_all.log"
```

## Citation

tbd
