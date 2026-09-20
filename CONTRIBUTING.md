# Contributing

FlyIsaac is experimental research/hobby code and currently targets Windows + The Binding of Isaac: Repentance+.

## Before changing runtime behavior

Please keep these invariants intact unless the change explicitly targets them:

- do not delete or overwrite `fly_agent/isaac/` blindly; current V3.x code still imports legacy modules
- preserve the DAgger teacher hotfix: human correction mode must not release WASD/arrow labels every frame
- preserve focus-pause behavior
- keep pixel-only and privileged-grid modes clearly distinguishable
- do not silently replace the user's `isaac_v3/dopamine_settings.py`
- do not describe grid-assisted results as pure biological vision
- do not claim current policy-residual learning is full MaleCNS synaptic plasticity

## Basic checks

From `fly_agent/`:

```powershell
python verify_isaac_v37.py
python test_tactical_grid_v37.py
python test_dopamine_preserved_v37.py
python test_anatomy_v37.py
python test_sensory_v37.py
python test_fast_loop_v37.py
python verify_door_memory_v371.py
python test_door_memory_v371.py
```

For a quick repository-wide syntax check:

```powershell
python -m compileall -q .
```

## Isaac bridge changes

If you change a Lua bridge, document which prefix it emits:

- `FLYAI|`
- `FLYROOM|`
- `FLYWORLD|`
- `FLYCOMBAT|`
- `FLYCOMBATV37|`

Use:

```powershell
python check_isaac_bridges_v371.py --seconds 30
```

during a live Isaac session.

## Pull requests

Keep PRs focused. In the description, state:

1. what changed,
2. which runtime mode is affected,
3. whether old datasets/checkpoints remain compatible,
4. whether a live Isaac test was performed or only synthetic/unit tests were run.
