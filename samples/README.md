# Test samples

## `jfk_10s_16k_mono.wav`

| Property | Value |
| --- | --- |
| Duration | 10.000 s (measured with `ffprobe`) |
| Format | PCM signed 16-bit LE, 16 kHz, mono |
| Size | 320,078 bytes |
| Source | [File:JFK inaugural address.ogg](https://commons.wikimedia.org/wiki/File:JFK_inaugural_address.ogg) on Wikimedia Commons (840.23 s full recording) |
| Extract | seconds 40.0–50.0 of the source |
| Licence | Public domain — a work of the United States federal government (17 U.S.C. § 105) |

Chosen because it is English speech, licence-clear, and 16 kHz mono matches Sarvam's
recommended input format for `/speech-to-text`.

Reproduce with:

```bash
curl -L -H "User-Agent: your-app/0.1 (you@example.com)" \
  "https://upload.wikimedia.org/wikipedia/commons/d/d5/JFK_inaugural_address.ogg" \
  -o jfk_full.ogg

ffmpeg -ss 40 -t 10 -i jfk_full.ogg -ac 1 -ar 16000 -c:a pcm_s16le \
  samples/jfk_10s_16k_mono.wav
```
