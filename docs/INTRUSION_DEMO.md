# Phase 2 — restricted-zone intrusion and incident evidence

This feature adds a **configurable zone** to the laptop camera and MP4 pipeline. It uses an existing YOLO person detection and tracked person bounding box; intrusion is a **geometric rule**, not a separately trained AI classifier.

## Run

From the repository root in PowerShell:

```powershell
git switch snehasich
git pull origin snehasich
python main.py --source 0 --device auto
```

In a second terminal:

```powershell
python -m streamlit run dashboard/app.py
```

A red rectangle labeled **RESTRICTED ZONE** appears on the video. For a reliable demo, begin **outside** the rectangle, then move into it. A tracked person's **bottom-center** is used to detect zone entry. On entry, the pipeline saves a **HIGH** severity `RESTRICTED_ZONE_INTRUSION` event and a JPEG image.

The screenshot is saved under `data/evidence/`, referenced in the event's JSON metadata under `evidence_file`. On the Streamlit dashboard, select **Alerts only**, open **Inspect Event Details**, and choose the intrusion event to display its snapshot.

The screenshot stays on the local laptop; nothing is uploaded to an external cloud service by this feature.

## Adjusting zone position

Edit `configs/default.yaml`:

```yaml
intrusion:
  enabled: true
  zone: [0.35, 0.25, 0.75, 0.85]
  cooldown_seconds: 20
```

The four values represent **left, top, right, bottom** as fractions of the camera width/height, each between 0 and 1. For example, `[0.50, 0.20, 0.95, 0.90]` places the rectangle on the right side of the video. Restart `main.py` after updating the configuration. Set `enabled: false` to disable intrusion monitoring.

The rule saves one event per **new entry**, not one per frame. A 20-second per-tracking-ID cooldown reduces repeated alerts. A person already inside the zone when surveillance starts is treated as an initial entry.

## Test

```powershell
python -m pytest tests/test_intrusion.py tests/test_evidence.py -v
```

## Important limitations

- Bounding-box location is not a precise foot position; detections can be imperfect.
- Tracking IDs may change after occlusion, causing duplicate entry alerts.
- The zone is a fixed screen rectangle, not a calibrated real-world boundary.
- Evidence images may contain identifiable people. Obtain consent, limit access, and periodically delete old evidence. The database cleanup button does not currently remove JPEG evidence files.
- False alarm rates and on-device performance must be evaluated with the actual examination laptop.

