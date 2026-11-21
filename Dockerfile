# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime

FROM runtime AS baseline
COPY selector.py /app/selector.py
WORKDIR /app
CMD ["python", "-B", "-c", "import selector; selector.WORKERS = 4; selector.main()"]

FROM runtime AS submission
COPY coverage_selector.py dialect_impact.py patch_impact.py source_impact.py /app/
COPY attribute_witnesses.py comparison_witnesses.py condition_witnesses.py /app/
COPY vendor/coverage.json.gz /app/coverage.json.gz
COPY vendor/dialect-index.json.gz /app/dialect-index.json.gz
COPY vendor/witnesses.json.gz /app/witnesses.json.gz
ADD vendor/sqlglot-eb11aec.tar.gz /app/sqlglot-source/
CMD ["python", "-B", "/app/coverage_selector.py", "--coverage", "/app/coverage.json.gz", "--source", "/app/sqlglot-source", "--dialect-index", "/app/dialect-index.json.gz", "--witnesses", "/app/witnesses.json.gz"]
