PYTHON ?= .venv/Scripts/python.exe
INPUT   ?= samples/jfk_10s_16k_mono.wav
OUTPUT  ?= output/dubbed.mp4
SRC     ?= en-IN
TGT     ?= hi-IN

.PHONY: help install test verify run dry-run clean-cache clean

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install Python dependencies into the active environment
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run the full pytest suite (no network access)
	$(PYTHON) -m pytest

verify:  ## M1 acceptance: real ASR call, then the identical call served from cache
	$(PYTHON) -m scripts.m1_verify --input $(INPUT)

verify-free:  ## Re-prove M1 from the warm cache, costing zero rupees
	$(PYTHON) -m scripts.m1_verify --input $(INPUT) --reuse-cache

run:  ## Run the pipeline end to end
	$(PYTHON) -m src.pipeline --input $(INPUT) --output $(OUTPUT) \
		--source-lang $(SRC) --target-lang $(TGT)

dry-run:  ## Run entirely from cache; fails loudly on any cache miss
	$(PYTHON) -m src.pipeline --input $(INPUT) --output $(OUTPUT) \
		--source-lang $(SRC) --target-lang $(TGT) --dry-run

clean-cache:  ## Delete every cached API response (the next run will spend credits)
	$(PYTHON) -c "from src.cache import DiskCache; print(DiskCache('cache').clear(), 'files removed')"

clean:  ## Remove generated outputs and Python caches
	rm -rf output/*.mp4 output/*.wav output/*.json .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
