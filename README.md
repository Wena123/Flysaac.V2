# FlyIsaac

**A Drosophila MaleCNS connectome controlling The Binding of Isaac: Repentance+.**

[![Python syntax](https://github.com/Wena123/Flysaac.V2/actions/workflows/python-syntax.yml/badge.svg)](https://github.com/Wena123/Flysaac.V2/actions/workflows/python-syntax.yml)

FlyIsaac is an experimental real-time control project that connects a large **Drosophila MaleCNS** connectome simulation to **The Binding of Isaac: Repentance+**.

The project intentionally mixes biologically derived connectivity with engineered perception, game-state bridges and small trainable decoders. The assisted mode is therefore **not** a claim that a fly brain naturally understands Isaac.

**Current release: V3.7.1**

![FlyIsaac V3.7 — Isaac gameplay beside the MaleCNS Anatomical Activity 2.0 monitor](https://github.com/Wena123/Flysaac.V2/blob/main/docs/images/flyisaac-v37-hero.webp?raw=1)

*Live V3.7 runtime: Isaac gameplay beside the MaleCNS Anatomical Activity 2.0 monitor, retina views, tactical grid and decoder outputs.*

## Highlights

- MaleCNS runtime through FlyBrain (~166k neurons in the current setup)
- pixel retina / motion / target processing
- descending-neuron (DN) traces
- trainable movement + shooting decoder
- 7×14 tactical neural grid
- legacy 12×20 decoder grid kept for compatibility
- hostile projectile tracking with:
  - current position
  - velocity
  - hitbox
  - +150 ms / +300 ms / +500 ms prediction
- engineered projectile-danger drive into LC10 / LPLC2 / LC4 pathways
- PAM/PPL-like dopamine signalling when matching annotations are available
- bounded online dopamine-gated action-bias memory
- door-memory anti-farming for repeated A↔B transitions
- ROOM_CLEAR de-duplication per room/run
- 5-minute same-room watchdog → hold **R** for 3 s → transient reset
- anatomical MaleCNS activity monitor
- DAgger human-correction workflow
- low-latency capture loop with stale-frame dropping

## Architecture

```text
Isaac pixels
    │
    ▼
retina / visual features
    │
    ▼
engineered visual sensory drive
    │
    ▼
┌─────────────────────────┐
│         MaleCNS         │
│  biological connectivity│
└────────────┬────────────┘
             │
             ▼
     descending neurons
             │
             ▼
      trainable decoder
             │
             ▼
   movement + shooting


Isaac Lua state
    │
    ├──► 12×20 compatibility grid ───► decoder
    │
    └──► 7×14 tactical grid ─────────► engineered MaleCNS sensory drive
             │
             └── projectile prediction / danger
```

### Pure vision vs assisted mode

`--pure-vision` uses the pixel-derived visual path without the privileged tactical grid.

`--grid-assist` additionally uses room/combat state from Isaac's Lua/log bridge. That information is **privileged game state**, not something inferred from the screen.

This distinction matters when interpreting results.

## Requirements

Known development environment:

- Windows 11
- Python 3.11 64-bit
- The Binding of Isaac: Repentance+
- NVIDIA GPU recommended
- CUDA 12-compatible driver recommended
- `flybrain[gpu]==0.1.0`

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Optional CUDA check:

```powershell
python -c "import cupy as cp; print('CUDA devices:', cp.cuda.runtime.getDeviceCount()); print('CuPy:', cp.__version__)"
```

## Quick start

Clone the repository and enter the runtime folder:

```powershell
git clone https://github.com/Wena123/Flysaac.V2.git
cd Flysaac.V2\fly_agent
```

Run current verification:

```powershell
python verify_isaac_v37.py
python verify_door_memory_v371.py
python test_door_memory_v371.py
```

Check Isaac bridge streams while the game is running:

```powershell
python check_isaac_bridges_v371.py
```

Typical workflow:

```powershell
python collect_isaac_v3.py --minutes 15
python train_decoder_v3.py
python play_isaac_v3.py --grid-assist
python dagger_isaac_v3.py
```

For the longer Windows command reference see:

```text
FlyIsaac_KOMENDY_V3.7.1.txt
```

## Runtime modes

Assisted mode:

```powershell
python play_isaac_v3.py --grid-assist
```

Pure visual mode:

```powershell
python play_isaac_v3.py --pure-vision
```

Pure vision without dopamine:

```powershell
python play_isaac_v3.py --pure-vision --no-dopamine
```

Grid decoder input without grid injection into MaleCNS:

```powershell
python play_isaac_v3.py --grid-assist --no-brain-grid
```

Dopamine neural signal without online policy-bias updates:

```powershell
python play_isaac_v3.py --grid-assist --no-dopamine-plasticity
```

Disable the neuron monitor:

```powershell
python play_isaac_v3.py --grid-assist --no-neuron-monitor
```

## Isaac bridge mods

Bundled under `isaac_mods/`:

```text
isaac_mods/
├── FlyAI_Combat_Bridge_V4/
└── FlyAI_V37_Projectile_Bridge/
```

The runtime also expects the legacy project streams:

```text
FLYAI|
FLYROOM|
FLYWORLD|
FLYCOMBAT|
FLYCOMBATV37|
```

The exact legacy core/room/world Lua source folders were not present in the V3.7/V3.7.1 release artifacts used to assemble this repository, so replacements were **not invented and presented as originals**. Keep the already-working legacy bridge mods installed.

Live check:

```powershell
python check_isaac_bridges_v371.py --seconds 30
```

## Dopamine

User-editable dopamine configuration:

```text
fly_agent/isaac_v3/dopamine_settings.py
```

The current V3.7.1 configuration intentionally keeps the custom event values used during development.

Persistent policy memory:

```text
fly_agent/checkpoints/isaac_dopamine_policy_v36.npz
```

Important: current persistent online learning is a **bounded policy residual / action-bias mechanism**. It is not general persistent synaptic plasticity across the whole MaleCNS connectome.

## Door memory / anti-farming

V3.7.1 stores traversed room connections as undirected edges:

```text
A <-> B
```

The first crossing can receive `NEW_ROOM` dopamine credit. Repeated A→B→A→B movement through the same connection does not repeatedly farm that reward.

`ROOM_CLEAR` is also de-duplicated once per room per run.

## Projectile vision

The V3.7 projectile bridge reports hostile projectile position, velocity and hitbox data. Python builds tactical predictions for:

```text
NOW
+150 ms
+300 ms
+500 ms
```

The goal is to provide a spatial danger representation rather than only the nearest-projectile distance.

## Performance

Performance settings live in:

```text
fly_agent/isaac_v3/performance_settings.py
```

Current design principle:

> **freshest frame wins**

If capture runs faster than the MaleCNS loop, stale frames are discarded instead of building a latency queue.

## Same-room watchdog

If the agent stays in the same room for approximately five minutes:

1. controls are released,
2. `R` is held for three seconds,
3. the run restarts,
4. transient MaleCNS / retina / temporal state is reset,
5. learned decoder files and persistent dopamine policy memory remain intact.

Settings:

```text
fly_agent/isaac_v3/stuck_settings.py
```

Disable:

```powershell
python play_isaac_v3.py --grid-assist --no-stuck-watchdog
```

## Repository layout

```text
Flysaac.V2/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── FlyIsaac_KOMENDY_V3.7.1.txt
├── isaac_mods/
└── fly_agent/
    ├── play_isaac_v3.py
    ├── collect_isaac_v3.py
    ├── dagger_isaac_v3.py
    ├── train_decoder_v3.py
    ├── isaac/        # legacy modules still used by current runtime
    ├── isaac_v3/     # current V3.x runtime
    ├── datasets/
    └── checkpoints/
```

## Cleanup

The repository includes:

```powershell
python cleanup_legacy_v371.py
```

It removes only a defined set of obsolete diagnostic/test files plus Python caches. It deliberately leaves core source, datasets, checkpoints, mods and dopamine settings untouched.

## Research status and limitations

This is an experimental system, not a validated model of fly cognition.

In particular:

- MaleCNS connectivity is biologically derived.
- visual preprocessing is engineered.
- the tactical grid is privileged game-state input.
- grid-to-MaleCNS routing is engineered.
- the PAM/PPL valence mapping is an engineered computational interpretation.
- the current persistent online learner changes bounded action biases, not the full connectome.
- synthetic tests do not prove that every live Isaac bridge or every expected cell annotation is present on a given machine.

Keeping those distinctions explicit makes pixel-only and assisted experiments easier to compare honestly.

## Version history

See [CHANGELOG.md](CHANGELOG.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
