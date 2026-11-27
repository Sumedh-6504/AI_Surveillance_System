"""Policy for recording useful detection snapshots without flooding the event feed."""
from collections import Counter


def detection_signature(detections):
    """Stable, order-independent summary of visible object classes."""
    return tuple(sorted(Counter(d.get("class", "unknown") for d in detections).items()))


def should_save_detection(detections, last_signature, last_saved_at, now,
                          change_cooldown=5.0, heartbeat_interval=60.0):
    """Record a new scene after cooldown, or a periodic unchanged-scene snapshot."""
    if not detections:
        return False
    elapsed = now - last_saved_at
    if elapsed < change_cooldown:
        return False
    signature = detection_signature(detections)
    return signature != last_signature or elapsed >= heartbeat_interval
