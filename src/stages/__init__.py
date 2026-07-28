"""
Pipeline stages, each with a clean input/output contract and independently runnable.

Delivery order (see SPEC.md):
    demux.py    M2  ffmpeg -> 16 kHz mono WAV
    asr.py      M2  Saaras v3 -> segments.json with stitched timestamps
    translate.py M3 Sarvam-Translate -> per-segment target text
    tts.py      M4  Bulbul v3 + closed-loop duration fitting
    assemble.py M5  place segments on a silent timeline
    mux.py      M5  remux dubbed audio onto the original video
"""
