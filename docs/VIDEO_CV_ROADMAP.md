# Video Parsing and CV Roadmap

## What We Need to Extract
- Hand state: visible cards, elixir estimate, selected card.
- Arena entities: unit/tower locations, approximate health, team identity.
- Event stream: deployment actions and timestamps.

## Current Baseline (Implemented)
- `src/crbot/vision/state_extractor.py`
- `scripts/extract_video_states.py`
- `src/crbot/vision/tracking.py`
- `scripts/track_video_states.py`
- `src/crbot/vision/annotation_queue.py`
- `scripts/build_video_annotation_queue.py`
- Heuristic OpenCV segmentation for blue/red blobs as provisional entities.
- Bootstrap multi-frame `track_id` assignment with confidence-aware frame filtering.
- Priority queue generation for manual annotation with label budget summary.

This is a bootstrap for data plumbing, not final detection quality.

## Recommended CV Stack (Next Phases)

### Phase 1: Robust UI Anchors
- Detect fixed UI regions (hand strip, arena bounds, timer/elixir widgets).
- Normalize coordinates for different resolutions and aspect ratios.
- Add confidence checks per frame (skip low-confidence frames).

### Phase 2: Action Label Recovery
- Detect tap-like events from:
  - hand card selection changes
  - sudden unit spawn signatures
- Convert inferred events into `actions.jsonl` with confidence.

### Phase 3: Entity Detection Model
- Train a lightweight detector (e.g., YOLO-n/s) on annotated frames:
  - classes: `own_troop`, `enemy_troop`, `tower`, `spell_effect`
- Add temporal smoothing/tracking (SORT/ByteTrack or optical-flow linking).

### Phase 4: State Estimation
- Map tracked entities to simulator/grid features.
- Estimate hidden variables (cooldowns/elixir) with sequence model or filters.
- Emit policy-ready low-res features + quality/confidence masks.

## Data Labeling Strategy
- Start with a small high-quality labeled set (your own sessions).
- Use pseudo-labeling on replay/TV videos with human spot-checking.
- Keep confidence scores and filter low-confidence samples from BC training.

## Practical Guidance
- Use direct tap logs as ground truth whenever possible.
- Use video-derived labels as augmentation, not the sole source.
- Track label quality by source in dataset manifests for ablation.
