FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy scikit-learn falcon-challenge

WORKDIR /app
COPY chip/int4_gru.c chip/int4_gru.c
COPY chip/bench_int4_gru.py chip/bench_int4_gru.py
COPY falcon_submit/decoder.py falcon_submit/decoder.py
COPY falcon_submit/h1_int4_bundle.pkl falcon_submit/h1_int4_bundle.pkl

RUN gcc -O3 -shared -fPIC chip/int4_gru.c -o falcon_submit/libint4_gru.so

ENV PREDICTION_PATH="/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL="/tmp/submission.pkl"
ENV GT_PATH="/tmp/ground_truth.pkl"
ENV EVALUATION_LOC="remote"
ENV EVAL_DATA_PATH="/dataset/evaluation_data"
ENV SPLIT="h1"
ENV PHASE="minival"
ENV BATCH_SIZE="1"

CMD ["/bin/bash", "-c", "python falcon_submit/decoder.py --mode eval --evaluation $EVALUATION_LOC --model-path falcon_submit/h1_int4_bundle.pkl --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]
