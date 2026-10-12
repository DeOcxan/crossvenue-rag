CONFIG ?= configs/default.yaml
# The Markdown that lint-docs and lint-links check (docs/style.md, Checks).
DOCS   ?= README.md docs evals
UV     ?= uv

.PHONY: help install db db-down fetch ingest index eval select select-finalize label round2-sample relabel kappa adjudicate relabel-d23 retrieve stats test lint lint-docs lint-links fmt check check-config validate-evals ask serve reproduce figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv from uv.lock with the dev group and the `figures` extra (mypy checks scripts/, which import matplotlib)
	$(UV) sync --locked --group dev --extra figures

db:  ## Start Postgres + pgvector and wait for it
	docker compose up -d
	@until docker compose exec -T postgres pg_isready -U crossvenue -d crossvenue >/dev/null 2>&1; \
	  do printf '.'; sleep 1; done; echo ' postgres ready'

db-down:  ## Stop Postgres (keeps the volume)
	docker compose down

fetch:  ## Download the sources data/manifest.csv records and verify every hash
	$(UV) run crossvenue fetch --config $(CONFIG)

ingest:  ## Normalize and chunk the corpus
	$(UV) run crossvenue ingest --config $(CONFIG)

index: db  ## Build the BM25 and pgvector indexes
	$(UV) run crossvenue index --config $(CONFIG)

stats:  ## Print corpus statistics
	$(UV) run crossvenue stats --config $(CONFIG)

retrieve:  ## Ad-hoc query: make retrieve Q="what is the funding interval" M=all
	$(UV) run crossvenue retrieve "$(Q)" --method "$(or $(M),hybrid)" --config $(CONFIG)

eval:  ## Run the six-method ablation into results/<timestamp>/
	$(UV) run crossvenue eval --config $(CONFIG)

select:  ## Choose the eval set from the candidate pool
	$(UV) run crossvenue select --config $(CONFIG)

select-finalize:  ## Write evals/questions.jsonl from the accepted candidates
	$(UV) run crossvenue select --config $(CONFIG) --finalize

label:  ## Mark gold chunks for eval questions (round 1)
	$(UV) run crossvenue label --config $(CONFIG)

round2-sample:  ## Draw the fixed, stratified round-2 subsample into evals/selection/round2_sample.txt
	$(UV) run crossvenue round-sample --config $(CONFIG) --out evals/selection/round2_sample.txt

relabel:  ## Re-annotate the fixed round-2 subsample (evals/selection/round2_sample.txt), for intra-rater agreement
	$(UV) run crossvenue label --config $(CONFIG) --round 2 --sample-file evals/selection/round2_sample.txt

kappa:  ## Cohen's kappa between annotation rounds 1 and 2
	$(UV) run crossvenue kappa --config $(CONFIG)

adjudicate:  ## Decide the pairs annotation rounds 1 and 2 disagree on, under the amended criterion (methodology 5a)
	$(UV) run crossvenue label --config $(CONFIG) --adjudicate

relabel-d23:  ## Judge the three questions of deviation D23 again, with the answer chunks their pools missed
	$(UV) run crossvenue label --config $(CONFIG) --relabel cv-0106 cv-0115 cv-0121 --add-chunks evals/selection/d23_answer_chunks.jsonl

validate-evals:  ## Validate every dataset under evals/datasets/
	$(UV) run crossvenue validate-evals --config $(CONFIG)

ask:  ## Answer a question: make ask Q="how long is a binance ip ban"
	$(UV) run crossvenue ask "$(Q)" --config $(CONFIG)

serve:  ## Serve POST /answer and GET /health on 127.0.0.1:8000
	$(UV) run uvicorn crossvenue_rag.service.app:app_factory --factory --host 127.0.0.1 --port 8000

reproduce:  ## Check the numbers of README, findings, methodology, design, the ADRs, the three reference pages, the gold-set datasheet and the domain research against their sources, then run the full mutation test (not in CI)
	$(UV) run crossvenue reproduce
	$(UV) run pytest -m slow tests/reporting/test_mutations.py

figures:  ## Draw docs/figures/*.svg (embedded) and *.png (print copies, not committed) from saved runs (matplotlib, the `figures` extra)
	$(UV) run --extra figures python scripts/figures.py

check-config:  ## Validate config and report which artifacts exist
	$(UV) run crossvenue check --config $(CONFIG)

test:  ## Run the test suite without the slow full mutation test
	$(UV) run pytest

check: lint  ## Lint, the whole test suite with the slow full mutation test, and reproduce
	$(UV) run pytest -m ""
	$(UV) run crossvenue reproduce

lint: lint-docs lint-links  ## Lint code, types, import contracts, prose, links and Markdown formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy
	$(UV) run lint-imports
	$(UV) run pytest -q tests/test_repo_hygiene.py tests/test_code_hygiene.py

# Vale exits non-zero on errors only; any alert at warning level fails here.
lint-docs:  ## Lint prose in README.md, docs and evals with Vale (rules: docs/style.md)
	@out=$$(vale --output=line $(DOCS)); status=$$?; \
	  if [ -n "$$out" ]; then echo "$$out"; exit 1; fi; exit $$status

lint-links:  ## Check every relative link and heading anchor in README.md, docs and evals, offline
	lychee --offline --include-fragments --no-progress $(DOCS)

fmt:  ## Apply formatting and autofixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

clean:  ## Remove derived corpus data (keeps data/raw, the manifest and every run in results/)
	rm -rf data/processed data/index
