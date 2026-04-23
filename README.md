# Clash Royale RL Bot (Starter)

This repository is a practical starter stack for a Clash Royale research bot:

- Abstract simulator environment (`gymnasium` API)
- Trajectory dataset format for imitation learning
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

Train behavior cloning on the generated trajectories:

```powershell
python scripts/train_bc.py --data-dir data/sim_random --epochs 8 --out checkpoints/bc_sim.pt
```

Train PPO baseline on simulator:

```powershell
python scripts/train_ppo.py --timesteps 200000 --out checkpoints/ppo_sim
```

## Layout

```text
configs/                         config files
scripts/                         entrypoint scripts
src/crbot/sim/                   abstract simulator env
src/crbot/data/                  trajectory schema + io
src/crbot/models/                policy models
src/crbot/emulator/              BlueStacks + adb stubs
tests/unit/                      lightweight tests
```

## Important notes

- This is not a full-fidelity Clash Royale simulator yet; it is an RL research harness.
- Real-game automation may violate Supercell terms and can risk account bans.
- Use a throwaway account and isolate experimentation.

## Immediate next tasks

1. Replace heuristic `CrLikeSimEnv` combat with card-specific dynamics.
2. Add visual state extraction from emulator screenshots.
3. Add action masking for invalid placements and unavailable cards.
4. Replace random data collection with human/heuristic trajectories.

