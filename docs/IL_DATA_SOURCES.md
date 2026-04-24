# Imitation Data Sources

## Goal
Use multiple demonstration sources in one BC pipeline:
- Your own direct gameplay sessions
- Replay/video sessions (including Clash Royale TV)

## Source Types

### 1) Direct Play Sessions (Best)
- Collected with `scripts/record_emulator_session.py`
- Converted with `scripts/build_tap_bc_dataset.py`
- Has real touch events, so action labels are naturally available.

Output:
- `data/il_tap/*.npz`

### 2) Replay/Video Sessions (Needs Labels)
- Use `scripts/build_video_bc_dataset.py`
- Each session folder requires:
  - `frames.jsonl`
  - `actions.jsonl`

`actions.jsonl` rows:
- Preferred: `{ "timestamp": 12.34, "action": 173 }`
- Alternative: `{ "timestamp": 12.34, "slot": 1, "grid_x": 3, "grid_y": 9 }`

Important:
- Clash Royale TV videos do **not** include input logs, so you need a label
  source (manual annotation or CV model) to produce `actions.jsonl`.

Helpful workflow:
1. Extract states: `scripts/extract_video_states.py`
2. Track entities: `scripts/track_video_states.py`
3. Build label queue/budget: `scripts/build_video_annotation_queue.py`
4. Label top-priority frames/windows first.

Use shared card classes from `configs/cards_registry.yaml` when defining
label taxonomy so CV classes map cleanly to simulator card IDs.

You can bootstrap full-card coverage by syncing official card lists into
reviewable registry stubs:
- `scripts/sync_cards_from_api.py` (JSON input or live API)

Output:
- `data/il_video/*.npz`

## Merge Multiple Sources
- Use `scripts/merge_il_datasets.py` to combine episode files.

Example:
```powershell
python scripts/merge_il_datasets.py --inputs data/il_tap data/il_video --out data/il_merged --prefix-with-source
```

Then train:
```powershell
python scripts/train_bc.py --data-dir data/il_merged --epochs 8 --out checkpoints/bc_merged.pt
```

Split merged IL data into train/val/test:
```powershell
python scripts/split_il_dataset.py --data-dir data/il_merged --out data/il_split --group-by-source
python scripts/train_bc.py --data-dir data/il_split/train --epochs 8 --out checkpoints/bc_merged.pt
```

## Recommended Data Priority
1. Your own direct-play sessions (highest trust labels).
2. Replays/videos with high-quality labels.
3. Weakly-labeled video data only after strong filtering.
