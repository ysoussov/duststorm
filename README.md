# Duststorm Rescue GIF

Usage

1. Activate venv:
   source .venv/bin/activate
2. Install deps:
   pip install -r requirements.txt
3. Place your photos into `inputs/` as `person1.jpg` (the cactus clinger) and `person2.jpg` (the rescuer).
4. Run:
   python scripts/make_gif.py
5. Find the output at `outputs/duststorm.gif`.

Notes

- If OpenCV is not available, the script falls back to a center crop and still works.
- You can re-run after replacing the input photos; the output will be over-written.
- MP4 with audio: set optional env vars before running to control SFX:
  - WIND_SFX, WHOOSH_SFX, IMPACT_SFX: paths to your WAV/MP3 files
  - WIND_OFFSET, WHOOSH_OFFSET, IMPACT_OFFSET (seconds)
  - WIND_VOL, WHOOSH_VOL, IMPACT_VOL (e.g., 0.6, 1.2)
  Example:
    WIND_SFX=inputs/wind.wav WHOOSH_OFFSET=2.8 IMPACT_OFFSET=7.4 python scripts/make_gif.py
