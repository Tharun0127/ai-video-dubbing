PYTHON ?= .venv/Scripts/python.exe
INPUT   ?= samples/jfk_10s_16k_mono.wav
OUTPUT  ?= output/dubbed.mp4
SRC     ?= en-IN
TGT     ?= hi-IN

CLIP    ?= samples/test_clip.mp4

.PHONY: help install test verify verify-free verify-m2 test-api test-api-free \
        demux asr translate tts assemble mux qc dub dub-nofit run dry-run \
        cold-run variance-probe clean-cache clean

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

translate:  ## M3: translate every segment, asserting the mapping survives
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage translate

tts:  ## M4: synthesise every segment under the closed-loop duration fit
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage tts

assemble:  ## M5: place every clip on one silent timeline the length of the source
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage assemble

mux:  ## M5: remux the dubbed audio onto the original video (-c:v copy)
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage mux

qc:  ## M6: back-transcribe the dub, score it, and write qc_report.md
	$(PYTHON) -m src.pipeline --input $(CLIP) --stage qc

dub:  ## Run all seven stages on $(CLIP) end to end
	$(PYTHON) -m src.pipeline --input $(CLIP) --output $(OUTPUT) \
		--source-lang $(SRC) --target-lang $(TGT)

dub-nofit:  ## Same, with duration fitting disabled -- the audible before/after
	$(PYTHON) -m src.pipeline --input $(CLIP) --output output/dubbed_nofit.mp4 \
		--source-lang $(SRC) --target-lang $(TGT) --no-fit

run:  ## Run the pipeline end to end on $(INPUT)
	$(PYTHON) -m src.pipeline --input $(INPUT) --output $(OUTPUT) \
		--source-lang $(SRC) --target-lang $(TGT)

dry-run:  ## Run entirely from cache; fails loudly on any cache miss
	$(PYTHON) -m src.pipeline --input $(CLIP) --output $(OUTPUT) \
		--source-lang $(SRC) --target-lang $(TGT) --dry-run

cold-run:  ## Measure a true cold-cache end-to-end run (SPENDS CREDITS: ~Rs 2.2)
	$(PYTHON) -m src.pipeline --input $(CLIP) --output coldrun/dubbed.mp4 \
		--source-lang $(SRC) --target-lang $(TGT) --cache-dir coldrun/cache

variance-probe:  ## Measure Bulbul's run-to-run duration variance (SPENDS CREDITS: ~Rs 0.75)
	$(PYTHON) -m scripts.tts_variance_probe --repeats 3

clean-cache:  ## Delete every cached API response (the next run will spend credits)
	$(PYTHON) -c "from src.cache import DiskCache; print(DiskCache('cache').clear(), 'files removed')"

clean:  ## Remove generated outputs and Python caches
	rm -rf output/*.mp4 output/*.wav output/*.json output/*.md \
		output/audio_segments output/chunks output/qc_slices coldrun .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
