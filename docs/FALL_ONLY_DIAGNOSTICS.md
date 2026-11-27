# Debugging fall video clips without restricted-zone alerts

A `RESTRICTED_ZONE_INTRUSION` event does **not** mean a fall was identified.
Both modules run independently. Red `FALL` text already embedded in a recorded
video does not count as a model prediction.

## Fall-only examination mode

After pulling `snehasich` and ensuring `models/yolov8n-pose.pt` exists:

```powershell
python main.py --source "videos/fall_test.mp4" --device auto --mode fall --debug-fall
```

This forces fall detection **on** and intrusion detection **off**, regardless
of the default YAML flags, for this run only. It does not modify your YAML.
The dashboard continues to use the same SQLite database.

When the pose model is available, you should see
`Pose-based fall detector enabled on cpu` or `... on 0`.
Without pose weights, fall-only mode prints a clear error and does not start
the pipeline.

The terminal will periodically print `[FALL DEBUG]` diagnostics including:
- `active_tracks`: current tracker IDs
- `poses`: pose model detections
- `IoU`: overlap between pose and detector/tracker bounding boxes
- `posture`: upright, horizontal, other, or unknown
- `hip_drop`: normalized hip movement since an upright baseline
- `candidate`: start of a sustained horizontal posture
- `alerted`: whether an incident was already produced for this track

A successful fall event will have `rule_name: SUSPECTED_FALL` and a saved
`fall_<id>.jpg` image in `data/evidence/`. To see it in Streamlit, select
**Alerts only** -> **Inspect Event Details** and choose the matching event.

If you see `poses=0` or `posture=unknown`, pose estimation is failing on
those frames. If `hip_drop` remains small, posture changes are not meeting
the fall heuristic. If track IDs change between standing and falling, the
current fall state machine loses the earlier upright baseline.

Use a clip where the person is completely visible while standing, and remains
visible through the falling and lying portion. Test several normal activities,
including sitting and bending, as negative examples; do **not** lower the
threshold merely to force a single demonstration video to trigger.

## Return to intrusion monitoring

```powershell
python main.py --source 0 --mode intrusion --device auto
```

`--mode all` (the default) uses the feature flags in `configs/default.yaml`.
You can also clear earlier demo events through normal database cleanup or
filter them by event rule; old intrusion events do not disappear automatically.

## Tests

```powershell
python -m pytest tests/test_fall_pose.py tests/test_fall_demo_mode.py -v
```

Do not present the word `FALL` overlaid by the video as evidence that the model
detected anything; use an actual `SUSPECTED_FALL` SQLite event as the proof.


## 2026-11-27: Fix for object-track dropouts

A real test log showed that the YOLO pose model found a standing person
and then horizontal pose, while the regular object detector repeatedly returned
zero tracks. Fall monitoring now runs pose estimation **even when there are no
object tracks**, and uses the one visible pose to continue the single-person
temporal history. Such alerts use `track_id=-1` (unknown tracking ID) but
still have an evidence screenshot. The system still requires the posture
transition and sustained confirmation; a video label saying `FALL` is not
evidence of a detected event.

To test:

```powershell
git pull origin snehasich
python -m pytest tests/test_fall_pose.py tests/test_fall_tracking_handoff.py tests/test_fall_demo_mode.py -v
python main.py --source "videos/fall_test.mp4" --device auto --mode fall --debug-fall
```

The debug output will show `pose-only posture=...` where the pose model finds
a person but object tracking does not. The alert requires continuously observed
horizontal evidence; if the person disappears or leaves the video too soon,
`SUSPECTED_FALL` should not be saved.

The single-person fallback is only for unambiguous frames. If multiple people
are visible, the detector requires a valid object-track match. Camera cuts and
different people entering the same scene can still confuse temporal identity;
review actual video performance before using this system for real monitoring.
