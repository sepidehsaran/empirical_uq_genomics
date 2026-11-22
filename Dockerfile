# Reproduce every figure in the paper from the released data bundle.
#
#   docker build -t uq-genomics .
#   docker run --rm -v "$PWD/figure_data_bundle:/data:ro" -v "$PWD/cluster_preds:/preds:ro" \
#       -v "$PWD/out:/out" uq-genomics
#
# That regenerates all figures into ./out on the host. To poke around instead:
#   docker run --rm -it -p 8888:8888 -v "$PWD/figure_data_bundle:/data:ro" \
#       -v "$PWD/out:/out" uq-genomics jupyter lab --ip=0.0.0.0 --allow-root --no-browser
#
# This image deliberately does NOT install TensorFlow. Retraining needs a GPU and a
# CUDA base image; see docs/REPRODUCING.md. The figure path is CPU-only.
FROM python:3.10-slim

# fonts: matplotlib falls back to a bitmap font without them and the PDFs look wrong
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-figures.txt .
RUN pip install --no-cache-dir -r requirements-figures.txt jupyterlab

COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src
# mount the released bundle read-only at /data
ENV UQFIG_BUNDLE=/data
ENV UQFIG_PREDS=/preds
ENV MPLBACKEND=Agg

CMD ["./scripts/make_all_figures.sh", "/out"]
