# Clash Royale RL Bot (Starter)

This repository is a practical starter stack for a Clash Royale research bot:

- Abstract simulator environment (`gymnasium` API)
- Trajectory dataset format for imitation learning
- Emulator session recorder (frames + touch events)
- Behavior cloning (BC) baseline trainer in PyTorch
- PPO trainer scaffold (via Stable-Baselines3)
- BlueStacks/ADB interface stubs for real-game integration

## Quick Start (2 minutes)

Install once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,rl]"
```

Run a minimal end-to-end baseline:

```powershell
python scripts/collect_random_sim_data.py --episodes 50 --out data/sim_quick
python scripts/train_bc.py --data-dir data/sim_quick --epochs 3 --out checkpoints/bc_quick.pt
python scripts/eval_policy.py --policy bc --checkpoint checkpoints/bc_quick.pt --episodes 10 --out data/eval/bc_quick.json
```

If you only want to sanity-check setup:

```powershell
pytest -q
```

## Common Workflows

### Simulator RL/IL baseline

Collect simulator trajectories:

```powershell
python scripts/collect_random_sim_data.py --episodes 200 --out data/sim_random
```

Train BC:

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --out checkpoints/bc_sim.pt
```

Train mask-aware BC (if `action_masks` exist):

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --mask-actions --out checkpoints/bc_sim_masked.pt
```

Train PPO:

```powershell
python scripts/train_ppo.py --timesteps 200000 --out checkpoints/ppo_sim
python scripts/train_ppo.py --timesteps 200000 --out checkpoints/ppo_sim_masked --mask-actions
```

### Video-to-state pipeline

Extract baseline entities from replay video:

```powershell
python scripts/extract_video_states.py --video replays/match1.mp4 --out data/video_states/match1_states.jsonl --stride 3 --min-frame-confidence 0.2 --skip-low-confidence
```

Assign stable track IDs across frames:

```powershell
python scripts/track_video_states.py --in data/video_states/match1_states.jsonl --out data/video_states/match1_tracks.jsonl
```

Build prioritized manual-label queue + budget summary:

```powershell
python scripts/build_video_annotation_queue.py --states data/video_states/match1_tracks.jsonl --out-queue data/video_states/match1_annotation_queue.jsonl --out-summary data/video_states/match1_annotation_summary.json --top-k 300
```

Export top queued frames for fast manual labeling:

```powershell
python scripts/export_video_annotation_frames.py --video replays/match1.mp4 --queue data/video_states/match1_annotation_queue.jsonl --out-dir data/video_states/match1_label_frames --top-k 300
```

Build BC dataset from labeled video sessions:

```powershell
python scripts/build_video_bc_dataset.py --sessions-dir video_sessions --out data/il_video
```

### Emulator recording pipeline

Record a session:

```powershell
python scripts/record_emulator_session.py --duration 180 --fps 3 --out recordings
```

Convert recording to BC episodes:

```powershell
python scripts/build_tap_bc_dataset.py --recordings-dir recordings --out data/il_tap
```

### Dataset ops + evaluation

Merge/split IL datasets:

```powershell
python scripts/merge_il_datasets.py --inputs data/il_tap data/il_video --out data/il_merged --prefix-with-source
python scripts/split_il_dataset.py --data-dir data/il_merged --out data/il_split --group-by-source
```

Evaluate and compare runs:

```powershell
python scripts/eval_policy.py --policy random --episodes 20 --out data/eval/random_eval.json
python scripts/compare_eval_runs.py --glob "data/eval/*.json" --sort-by mean_reward
```

Visualize sim episodes:

```powershell
python scripts/visualize_sim_episode.py --steps 240 --fps 10 --out data/sim_viz/episode.gif
python scripts/visualize_sim_episode.py --steps 240 --fps 10 --out data/sim_viz/episode.gif --metrics-out data/sim_viz/episode_metrics.csv
```

Benchmark sim throughput:

```powershell
python scripts/benchmark_sim_throughput.py --episodes 30 --max-steps 300 --out data/eval/sim_throughput.json
```

## Layout

```text
configs/                         config files
scripts/                         entrypoint scripts
src/crbot/sim/                   abstract simulator env
src/crbot/data/                  trajectory schema + io
src/crbot/models/                policy models
src/crbot/emulator/              BlueStacks + adb stubs
src/crbot/recording/             real-play recording + label building
tests/unit/                      lightweight tests
```

Card metadata registry:
- `configs/cards_registry.yaml`
- Used by simulator card semantics and intended as shared source for CV labels.
- Supports archetype inheritance and per-card overrides for scaling to full roster metadata.
- Optional `extra.sim_profile` supports per-card simulator tuning (range, cooldown, splash, projectile speed, move interval, air override).
- Optional `extra.sim_profile` also supports targeting behavior overrides (e.g. `target_preference: high_hp`).
- Optional `extra.deploy_profile` supports placement constraints (e.g. `requires_bridge_lane_deploy`, `allow_enemy_side_deploy`).
- Optional `extra.spell_profile` supports spell-specific tuning (e.g. `direct_damage_mult`, `self_damage_mult`, `king_share`, `ignore_board_factor`, `ignore_range_factor`).

Sync official card list into registry stubs (dry-run by default):

```powershell
python scripts/sync_cards_from_api.py --from-json data/cards_api/cards.json
python scripts/sync_cards_from_api.py --from-json data/cards_api/cards.json --apply
```

Or fetch directly from API (requires token env var):

```powershell
$env:CR_API_TOKEN="YOUR_TOKEN"
python scripts/sync_cards_from_api.py --apply
```

Tip: if `--from-json` fails with file-not-found, fetch once and cache payload:

```powershell
$env:CR_API_TOKEN="YOUR_TOKEN"
python scripts/sync_cards_from_api.py --save-json data/cards_api/cards.json
python scripts/sync_cards_from_api.py --from-json data/cards_api/cards.json --apply
```

Review and bulk-fix pending stubs:

```powershell
python scripts/review_card_registry.py --summary --list --tag needs_review --limit 30
python scripts/review_card_registry.py --tag needs_review --set-archetype troop_any --remove-tag needs_review --add-tag reviewed --apply
```

Build prioritized review backlog with suggested archetypes:

```powershell
python scripts/build_card_registry_backlog.py --limit 500
```

If no `needs_review` cards exist yet, this command now auto-falls back to include
reviewed cards (disable with `--no-fallback-include-reviewed`).

Auto-apply safe backlog suggestions (keyword-based spells/buildings):

```powershell
python scripts/auto_review_card_registry.py
python scripts/auto_review_card_registry.py --apply
python scripts/build_card_registry_backlog.py --limit 500
python scripts/review_card_registry.py --summary --list --tag needs_review --limit 50
```

## Important notes

- This is not a full-fidelity Clash Royale simulator yet; it is an RL research harness.
- Real-game automation may violate Supercell terms and can risk account bans.
- Use a throwaway account and isolate experimentation.

## Immediate next tasks

1. Extend stateful unit dynamics with pathing, targeting priorities, and richer board interactions.
2. Add visual state extraction from emulator screenshots.
3. Integrate card metadata into combat and reward dynamics.
4. Replace random data collection with human/heuristic trajectories.

## High-Res State + Low-Res Obs

- The simulator now keeps an internal high-resolution state and separately
  builds policy observations from that state.
- By default, observations stay lightweight (`global`, `hand_ids`,
  `hand_costs`).
- You can optionally expose low-resolution unit density maps to the model via
  config:
  - `sim.observe_unit_density: true`
  - `sim.obs_grid_w`, `sim.obs_grid_h`

## Real-play data notes

- `record_emulator_session.py` reads touch events from `adb shell getevent -lt`.
- It captures frame images and touch logs into `recordings/session_*`.
- `build_tap_bc_dataset.py` reconstructs actions by pairing:
  - tap in hand slot, then
  - tap in arena within a short timeout.
- The resulting action encoding matches the simulator's discrete action format.

## Project workflow

- Roadmap: `docs/ROADMAP.md`
- Backlog: `docs/BACKLOG.md`
- Simulator status: `docs/SIM_STATUS.md`
- IL source guide: `docs/IL_DATA_SOURCES.md`
- Video CV roadmap: `docs/VIDEO_CV_ROADMAP.md`
- GitHub issue templates live under `.github/ISSUE_TEMPLATE/`

Bootstrap labels + starter issues (dry-run by default):

```powershell
python scripts/bootstrap_github_workflow.py
```

Apply remotely:

```powershell
$env:GITHUB_TOKEN="YOUR_TOKEN"
python scripts/bootstrap_github_workflow.py --apply
```

Re-running is idempotent by default (existing issue titles are skipped). Use
`--allow-duplicates` if you explicitly want new duplicates.

Sync those managed issues into a GitHub ProjectV2 board (create if missing):

```powershell
$env:GITHUB_TOKEN="YOUR_TOKEN"
python scripts/bootstrap_github_workflow.py --apply --sync-project --project-title "Clash Royale RL Bot"
```

Optional: set a project owner explicitly (user or org login) and status:

```powershell
python scripts/bootstrap_github_workflow.py --apply --sync-project --project-owner YOUR_LOGIN --project-title "Clash Royale RL Bot" --project-status "Todo"
```
