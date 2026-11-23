FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY fixtures/ ./fixtures/

RUN pip install --no-cache-dir -e "."

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/healthz')"

ENTRYPOINT ["fusion"]
CMD ["serve"]
