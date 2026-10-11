# Per-job pipeline image. Default base is the aarch64 vLLM image (CUDA + torch for DGX Spark / GB10);
# for x86_64 use --build-arg BASE=vllm/vllm-openai:v0.29.0, for CPU/API mode BASE=python:3.12-slim EXTRAS=.
ARG BASE=vllm/vllm-openai:v0.29.0-aarch64
FROM ${BASE}
ARG EXTRAS=gpu

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg rclone curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs \
    && npm install -g @plaud-ai/cli@0.3.15 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"

ENV PLAUDIT_CONFIG=/plaudit/plaudit.yaml
VOLUME ["/plaudit", "/root/.plaud", "/root/.config/rclone", "/root/.cache/huggingface"]
ENTRYPOINT ["plaudit"]
CMD ["run"]
