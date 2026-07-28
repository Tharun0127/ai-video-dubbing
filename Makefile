PYTHON ?= .venv/Scripts/python.exe
INPUT   ?= samples/jfk_10s_16k_mono.wav
OUTPUT  ?= output/dubbed.mp4
SRC     ?= en-IN
TGT     ?= hi-IN

CLIP    ?= samples/test_clip.mp4

.PHONY: help install test verify verify-free verify-m2 test-api test-api-free \
        demux asr run dry-run clean-cache clean

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

test-api:  ## M1 live API test on samples/test_audio.wav (one real call, then cached)
	$(PYTHON) -m src.test_m1_api

test-api-free:  ## Re-prove the M1 live API test from the warm cache, costing zero rupees
	$(PYTHON) -m src.test_m1_api --reuse-cache

demux:  ## M2: extract $(CLIP)'s audio to 16 kHz mono PCM WAV
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage demux

asr:  ## M2: chunk under the 30s cap, transcribe, stitch, and write segments.json
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage asr

verify-m2:  ## M2 acceptance: chunk plan, audio integrity, seams, timeline, cache
	$(PYTHON) -m scripts.m2_verify --input $(CLIP)

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
