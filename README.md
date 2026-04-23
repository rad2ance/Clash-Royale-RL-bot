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

Train behavior cloning on the generated trajectories:

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --out checkpoints/bc_sim.pt
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

1. Replace heuristic `CrLikeSimEnv` combat with card-specific dynamics.
2. Add visual state extraction from emulator screenshots.
3. Add richer card-type-aware placement rules (e.g. buildings constrained lanes and river constraints).
4. Replace random data collection with human/heuristic trajectories.

## Real-play data notes

- `record_emulator_session.py` reads touch events from `adb shell getevent -lt`.
- It captures frame images and touch logs into `recordings/session_*`.
- `build_tap_bc_dataset.py` reconstructs actions by pairing:
  - tap in hand slot, then
  - tap in arena within a short timeout.
- The resulting action encoding matches the simulator's discrete action format.
