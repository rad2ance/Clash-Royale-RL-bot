# Roadmap

## Milestone 1: Simulator Legality and Card Semantics
- Action masking integrated into data collection and PPO.
- River/bridge-aware placement rules.
- Card metadata registry (cost, type, targeting).
- Card-type-aware combat/reward shaping baseline.

## Milestone 2: Training Quality and Reproducibility
- Evaluation script for BC/PPO checkpoints.
- Run metrics logging to JSON.
- Baseline experiment presets in `configs/`.
- Trajectory schema supports optional action masks.

## Milestone 3: Emulator-to-Policy Loop
- Screenshot-to-state extractor stub with stable interface.
- Recorder pipeline emits aligned observations + actions.
- End-to-end smoke test from screenshot stream to policy action.

## Milestone 4: Stability and Release Hygiene
- CI for tests and linting.
- Better failure diagnostics for emulator connection and touch parsing.
- Documentation updates and contributor workflow polish.
