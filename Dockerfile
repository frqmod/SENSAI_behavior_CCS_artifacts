FROM rocker/r-ver:4.6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv libopenblas-dev libglpk40 \
    && rm -rf /var/lib/apt/lists/*

RUN install2.r --error --skipinstalled dplyr tidyr lme4 lmerTest emmeans seqHMM

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir pandas scikit-learn numpy scipy openai pydantic

ENV OPENBLAS_NUM_THREADS=1

WORKDIR /artifact
COPY . /artifact

CMD ["bash", "run_all.sh"]
