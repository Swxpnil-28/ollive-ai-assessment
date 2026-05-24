.PHONY: install run test eval clean docker-build docker-run lint

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest ruff

# ── Run ────────────────────────────────────────────────────────────────────────
run:
	streamlit run app.py

run-dev:
	STREAMLIT_SERVER_HEADLESS=false streamlit run app.py --server.runOnSave=true

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -x --tb=short

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval-hosted:
	python scripts/run_evals.py --model hosted

eval-oss:
	python scripts/run_evals.py --model oss

eval-all:
	python scripts/run_evals.py --model both

report:
	python reports/generate_report.py

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-run:
	docker compose up

docker-stop:
	docker compose down

# ── Code Quality ───────────────────────────────────────────────────────────────
lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf data/sessions/ data/traces.jsonl reports/*.csv
	echo "✅ Cleaned"
