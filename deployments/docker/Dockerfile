FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY tesseract_platform ./tesseract_platform
COPY integrations ./integrations
COPY contracts ./contracts

RUN pip install --upgrade pip && pip install .

RUN mkdir -p /app/var
EXPOSE 8088
ENTRYPOINT ["tesseract-platform"]
CMD ["serve"]
