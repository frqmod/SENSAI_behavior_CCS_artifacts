#!/usr/bin/env bash

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

echo -e "\n\n########## STAGE 01 classifier (with pre-fired data, shows 01->02 connection) ##########"
( cd 01_coding && python3 pipeline.py --assemble-only --input-dir input --output-csv data/sample_codes.csv --inputfeatures-csv inputfeatures_demo.csv --solves-csv data/sample_solves.csv && head -3 inputfeatures_demo.csv )

echo -e "\n\n########## STAGE 02 MHMM clustering (fresh fit) ##########"
( cd 02_mhmm && Rscript analysis.R fit )

echo -e "\n\n########## STAGE 02b fitted-model matrices ##########"
MODEL=$(cd 02_mhmm && ls -t mhmm_output/mhmm_*.rds | grep -v _session_data | head -1)
( cd 02_mhmm && Rscript analysis.R describe "$MODEL" )
