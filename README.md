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
python scripts/eval_policy.py --policy ppo --checkpoint checkpoints/ppo_sim.zip --episodes 20 --out data/eval/ppo_eval.json
python scripts/eval_policy.py --policy ppo-mask --checkpoint checkpoints/ppo_sim_masked.zip --episodes 20 --out data/eval/ppo_mask_eval.json
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
