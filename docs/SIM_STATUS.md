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
- Canonical card registry file at `configs/cards_registry.yaml` (sim-loaded).
- Registry supports archetype inheritance (`archetypes` + per-card overrides) for future full-card expansion.
- Registry supports optional per-card sim profile overrides via `extra.sim_profile`.
- Per-card target preference overrides are supported (`closest` / `low_hp` / `high_hp`).
- Per-card deploy/pathing overrides supported via registry:
  - `deploy_profile.requires_bridge_lane_deploy`
  - `deploy_profile.allow_enemy_side_deploy`
  - `sim_profile.can_cross_river_without_bridge`
- Per-card spell-effect overrides supported via registry:
  - `spell_profile.direct_damage_mult`
  - `spell_profile.self_damage_mult`
  - `spell_profile.king_share`
  - `spell_profile.ignore_board_factor` / `ignore_range_factor`
- API-sync + review tooling:
  - `scripts/sync_cards_from_api.py`
  - `scripts/review_card_registry.py`
- Deck-cycle hand replacement (deck queue instead of fully random redraw each play).
- Card-type-aware instant combat profile in `step()`.
- Stateful active units:
  - troop/building spawns
  - deterministic air-troop archetype tagging for "any"-targeting troops
  - TTL and HP decay
  - attack range + cooldown-based unit/tower attacks
  - optional projectile-travel mode with delayed impacts and splash radius
  - closest-target duel logic
  - bridge-aware river crossing pathing
  - air-vs-ground pathing split (air units can cross river off-bridge)
  - post-river lane-objective drift toward alive side towers
  - per-cell occupancy cap to prevent unrealistic unit stacking
  - layered occupancy (air and ground units no longer block each other by default)
  - lane-aware tower pressure targeting + princess fallback retargeting
  - per-unit lane target lock with retarget on objective destruction
  - archetype-aware projectile speed/splash profiles (in projectile mode)
  - movement speed tiers via per-unit move intervals
- Visual debugging:
  - `env.render()` RGB frames
  - GIF + CSV export via `scripts/visualize_sim_episode.py`
- Evaluation tooling:
  - `scripts/eval_policy.py`
  - `scripts/compare_eval_runs.py`
- Reward diagnostics:
  - per-step reward decomposition included in `info["reward_components"]`
  - explicit `info["reward_total"]` for sanity checks/debugging
- Vision extraction baseline upgrades:
  - explicit normalized UI anchors in extractor output
  - arena-scoped entity detection to reduce hand/UI false positives
  - per-frame confidence estimation and low-confidence gating
- Match-flow:
  - configurable regulation time
  - optional one-shot overtime extension when regulation ends with both kings alive
  - crown-score bookkeeping (princess/king objective tracking)
  - terminal winner resolution (`king_down` -> `crowns` -> HP tiebreak -> draw)

## Remaining High-Value Gaps
- True card mechanics:
  - per-card stats/abilities beyond coarse archetype templates
  - spell-specific areas/effects and timing windows
- Board simulation fidelity:
  - richer collision/avoidance beyond simple occupancy cap
  - richer projectile/splash geometry (partially card-archetype-aware)
  - partial tower-target lock/retarget behavior
- Pathing fidelity:
  - partial lane-objective pathing around bridges/towers
  - early bridge-approach alignment (units steer to bridge before river row)
  - partial movement speed realism (interval-based, not continuous velocity)
- Match rules:
  - basic single-phase overtime extension (simplified sudden-death proxy)
  - overtime/sudden death exact tie-break details
  - exact crown tie-break semantics and priority details
- Observation realism:
  - richer state channels (unit maps, explicit card cycle exposure, opponent hidden info model)
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
