# stage 02 - conversation-style MHMM

fits the mixture hidden Markov model that identified the Reactive / Proactive
conversation styles (2 clusters x 5 states, sequence length 4-22, EM with 10
random restarts; categorical emissions over the 7 query families)

- `analysis.R` - `fit` estimates the MHMM from `data/inputfeatures.csv` and writes auto-numbered `mhmm_NN.*` into `mhmm_output/`; `describe <model.rds>` prints the fitted initial/transition/emission matrices
- `data/inputfeatures.csv` - synthetic coded queries in the pipeline schema: `user_id, session_id, messagenum, task_code, module_num, solved`

## Usage

```bash
Rscript analysis.R fit
Rscript analysis.R describe mhmm_output/mhmm_01.rds  # inspect the fitted model
```
