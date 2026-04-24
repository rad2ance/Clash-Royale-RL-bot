# Clash Royale RL Bot (Starter)

This repository is a practical starter stack for a Clash Royale research bot:

- Abstract simulator environment (`gymnasium` API)
- Trajectory dataset format for imitation learning
- Emulator session recorder (frames + touch events)
- Behavior cloning (BC) baseline trainer in PyTorch
- PPO trainer scaffold (via Stable-Baselines3)
- BlueStacks/ADB interface stubs for real-game integration

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,rl]"
```

Generate random trajectories from the simulator:

```powershell
python scripts/collect_random_sim_data.py --episodes 200 --out data/sim_random
```

By default, random collection now samples only legal actions. To intentionally
sample from all actions (including illegal ones), pass:

```powershell
python scripts/collect_random_sim_data.py --episodes 200 --out data/sim_random --allow-illegal-actions
```

Collected simulator episodes now also persist per-step legal action masks
(`action_masks`) in each `.npz` file for offline analysis/training.

Train behavior cloning on the generated trajectories:

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --out checkpoints/bc_sim.pt
```

If your dataset includes `action_masks`, you can train BC with mask-aware
logits:

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --mask-actions --out checkpoints/bc_sim_masked.pt
```

Record a real BlueStacks play session (human demonstrations):

```powershell
python scripts/record_emulator_session.py --duration 180 --fps 3 --out recordings
```

Convert recorded sessions into BC episodes:

```powershell
python scripts/build_tap_bc_dataset.py --recordings-dir recordings --out data/il_tap
```

Build BC episodes from replay/video sessions with action labels:

```powershell
python scripts/build_video_bc_dataset.py --sessions-dir video_sessions --out data/il_video
```

Extract baseline CV entity states from raw replay video:

```powershell
python scripts/extract_video_states.py --video replays/match1.mp4 --out data/video_states/match1_states.jsonl --stride 3
```

Merge multiple IL sources:

```powershell
python scripts/merge_il_datasets.py --inputs data/il_tap data/il_video --out data/il_merged --prefix-with-source
```

Split IL data for training/eval:

```powershell
python scripts/split_il_dataset.py --data-dir data/il_merged --out data/il_split --group-by-source
```

Train PPO baseline on simulator:

```powershell
python scripts/train_ppo.py --timesteps 200000 --out checkpoints/ppo_sim
```

Train PPO with action masking (recommended):

```powershell
python scripts/train_ppo.py --timesteps 200000 --out checkpoints/ppo_sim_masked --mask-actions
```

Evaluate a policy and save JSON metrics:

```powershell
python scripts/eval_policy.py --policy random --episodes 20 --out data/eval/random_eval.json
python scripts/eval_policy.py --policy bc-mask --checkpoint checkpoints/bc_sim_masked.pt --episodes 20 --out data/eval/bc_mask_eval.json
python scripts/eval_policy.py --policy ppo --checkpoint checkpoints/ppo_sim.zip --episodes 20 --out data/eval/ppo_eval.json
python scripts/eval_policy.py --policy ppo-mask --checkpoint checkpoints/ppo_sim_masked.zip --episodes 20 --out data/eval/ppo_mask_eval.json
```

Compare multiple eval runs:

```powershell
python scripts/compare_eval_runs.py --glob "data/eval/*.json" --sort-by mean_reward
python scripts/compare_eval_runs.py --glob "data/eval/*.json" --sort-by win_rate --csv-out data/eval/compare.csv
```

Benchmark simulator throughput across resolution presets:

```powershell
python scripts/benchmark_sim_throughput.py --episodes 30 --max-steps 300 --out data/eval/sim_throughput.json
```

Visualize a simulator episode as GIF:

```powershell
python scripts/visualize_sim_episode.py --steps 240 --fps 10 --out data/sim_viz/episode.gif
```

Optional debug metrics CSV aligned to frames:

```powershell
python scripts/visualize_sim_episode.py --steps 240 --fps 10 --out data/sim_viz/episode.gif --metrics-out data/sim_viz/episode_metrics.csv
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
