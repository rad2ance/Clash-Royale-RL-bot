from crbot.vision import build_annotation_queue, summarize_annotation_budget


def test_annotation_queue_prioritizes_low_confidence_frames() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "frame_confidence": 0.95,
            "entities": [{"kind": "unit", "team": "own", "x_norm": 0.2, "y_norm": 0.3, "score": 0.9}],
        },
        {
            "timestamp": 0.1,
            "frame_confidence": 0.10,
            "entities": [{"kind": "unit", "team": "own", "x_norm": 0.2, "y_norm": 0.3, "score": 0.2}],
        },
    ]
    q = build_annotation_queue(rows, top_k=2)
    assert len(q) == 2
    assert q[0].timestamp == 0.1
    assert q[0].priority > q[1].priority


def test_annotation_queue_uses_new_track_signal() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "frame_confidence": 0.8,
            "entities": [{"track_id": 1, "kind": "unit", "team": "own", "x_norm": 0.2, "y_norm": 0.3, "score": 0.8}],
        },
        {
            "timestamp": 0.1,
            "frame_confidence": 0.8,
            "entities": [{"track_id": 2, "kind": "unit", "team": "own", "x_norm": 0.6, "y_norm": 0.3, "score": 0.8}],
        },
    ]
    q = build_annotation_queue(rows, top_k=2)
    assert q[0].new_tracks >= 1


def test_summarize_annotation_budget_scales_with_frame_count() -> None:
    rows = [{"timestamp": float(i), "entities": []} for i in range(1000)]
    s = summarize_annotation_budget(rows)
    assert s["n_frames"] == 1000
    rec = s["recommended_labels"]
    assert rec["quick_bootstrap"] >= 40
    assert rec["strong_baseline"] >= 150
    assert rec["robust_detector_seed"] >= 400
