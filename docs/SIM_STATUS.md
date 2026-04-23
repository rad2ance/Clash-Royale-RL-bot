# Simulator Status

## Implemented
- Discrete action space with legal action masking.
- High-resolution internal simulator state with separate observation builder.
- Placement legality rules:
  - elixir affordability
  - own-side deployment for non-spells
  - river-row restrictions
  - bridge-lane constraints for buildings
  - spells allowed across full arena
- Deterministic card metadata registry (`card_id -> type/cost/targeting`).
- Card-type-aware instant combat profile in `step()`.
- Stateful active units:
  - troop/building spawns
  - TTL and HP decay
  - attack range + cooldown-based unit/tower attacks
  - closest-target duel logic
  - bridge-aware river crossing pathing
- Visual debugging:
  - `env.render()` RGB frames
  - GIF + CSV export via `scripts/visualize_sim_episode.py`
- Evaluation tooling:
  - `scripts/eval_policy.py`
  - `scripts/compare_eval_runs.py`

## Remaining High-Value Gaps
- True card mechanics:
  - per-card stats/abilities beyond coarse archetype templates
  - spell-specific areas/effects and timing windows
- Board simulation fidelity:
  - unit collision and occupancy rules
  - projectile travel and splash geometry
  - exact tower targeting/retarget behavior
- Pathing fidelity:
  - lane/path graph around bridges and towers
  - more realistic movement speeds and stopping logic
- Match rules:
  - overtime/sudden death and tie-break details
  - crown logic and objective priorities
- Observation realism:
  - richer state channels (unit maps, card cycle, opponent hidden info model)
  - uncertainty/partial observability modeling

## Observation Architecture
- Internal simulation state is tracked at full simulator resolution and can be
  exported via `get_state_snapshot()`.
- Policy observations are generated separately via
  `build_observation_from_state(...)`.
- Optional low-resolution unit-density maps can be enabled with:
  - `observe_unit_density: true`
  - `obs_grid_w`, `obs_grid_h`
- Opponent model:
  - current enemy behavior is simplified stochastic pressure
  - needs scripted/learned adversary with deck-conditioned policy

## Practical Readiness
- Current sim is suitable for:
  - fast RL pipeline iteration
  - legality/masking research
  - debugging training/evaluation infrastructure
- Current sim is not yet suitable for:
  - claiming in-game tactical fidelity
  - direct real-match policy transfer without major domain adaptation
