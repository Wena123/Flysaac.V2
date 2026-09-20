# FlyIsaac / FlyBrain

Experimental project connecting a **Drosophila MaleCNS connectome simulation**
to **The Binding of Isaac: Repentance+**.

The goal is not to claim that a fruit-fly brain naturally understands Isaac.
The project deliberately combines biological connectivity with engineered
computer-vision, game-state bridges and small trainable action decoders so that
we can inspect what happens when a large connectome is placed inside a real-time
control loop.

**Current release:** V3.7.1

## What V3.7.1 contains

- full MaleCNS runtime through FlyBrain
- pixel-based retina / motion / target processing
- rich visual feature stream
- descending-neuron activity traces
- 7×14 **tactical neural grid**
- legacy 12×20 decoder grid retained for model compatibility
- hostile projectile tracking and short-horizon prediction:
  - NOW
  - +150 ms
  - +300 ms
  - +500 ms
- engineered projectile-danger drive into LC10 / LPLC2 / LC4 pathways
- PAM/PPL-like dopamine signalling when matching MaleCNS annotations exist
- bounded online dopamine-gated action-bias memory
- door memory to stop repeated A↔B doorway dopamine farming
- ROOM_CLEAR reward deduplication per room/run
- 5-minute same-room watchdog that holds **R for 3 seconds** and resets the
  transient runtime state
- enlarged anatomical MaleCNS activity monitor
- faster low-latency capture loop with stale-frame dropping
- DAgger human-correction workflow

## Architecture

```text
                         ┌──────────────────────┐
Isaac screen ───────────►│ pixel retina/vision  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                            visual sensory drive
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      MaleCNS         │
                         │  ~166k neurons       │
                         └──────────┬───────────┘
                                    │
                         descending neurons (DN)
                                    │
                                    ▼
                           trainable decoder
                                    │
                                    ▼
                         movement + shooting

Isaac Lua state ──► room/enemy/projectile bridges
                         │
                         ├──► 12×20 compatibility grid ─► decoder
                         │
                         └──► 7×14 tactical grid ───────► engineered
                                                        sensory drive
```

The tactical grid is **privileged game state**, not information inferred from
pixels. Results from `--grid-assist` therefore should not be described as a
pure visual/biological benchmark.

## Requirements

Known development setup:

- Windows 11
- Python 3.11 64-bit
- NVIDIA GPU recommended
- CUDA 12-compatible driver
- `flybrain[gpu]==0.1.0`
- The Binding of Isaac: Repentance+
- Steam version with Lua mods enabled

Install Python dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install "flybrain[gpu]==0.1.0" numpy opencv-python pygame mss pywin32 pydirectinput
```

Optional CUDA check:

```powershell
python -c "import cupy as cp; print('CUDA devices:', cp.cuda.runtime.getDeviceCount()); print('CuPy:', cp.__version__)"
```

## Project location

The current Windows development layout is:

```text
C:\Users\Admin\Documents\GitHub\FlyBrain\
├── .venv\
└── fly.ai\
    ├── README.md
    ├── isaac_mods\
    └── fly_agent\
```

Most commands must be run from:

```powershell
cd C:\Users\Admin\Documents\GitHub\FlyBrain\fly.ai\fly_agent
```

If Python says it cannot open `collect_isaac_v3.py`, `play_isaac_v3.py`, etc.,
check the current directory first.

## Isaac bridge mods

The repository contains `isaac_mods/`.

Source-confirmed bridge folders bundled with this release are:

```text
isaac_mods/
├── FlyAI_Combat_Bridge_V4/
└── FlyAI_V37_Projectile_Bridge/
```

The project also consumes the legacy streams:

```text
FLYAI|
FLYROOM|
FLYWORLD|
```

The exact legacy core/room/world Lua source folders were not present in the
V3.7/V3.7.1 release artifacts available when this package was assembled, so
they were **not fabricated and presented as originals**. Keep the already
working legacy bridge mods installed.

Check the live log streams with:

```powershell
python check_isaac_bridges_v371.py
```

Expected project prefixes are:

```text
FLYAI|
FLYROOM|
FLYWORLD|
FLYCOMBAT|
FLYCOMBATV37|
```

See `isaac_mods/README_MODS.md` for details.

## Verify the current release

From `fly_agent`:

```powershell
python verify_isaac_v37.py
python test_tactical_grid_v37.py
python test_dopamine_preserved_v37.py
python test_anatomy_v37.py
python test_sensory_v37.py
python test_fast_loop_v37.py

python verify_door_memory_v371.py
python test_door_memory_v371.py

python check_isaac_bridges_v371.py
```

These tests verify Python-side behavior and synthetic integration. They do not
replace a live Isaac + Lua bridge test.

## Typical workflow

Record human demonstrations:

```powershell
python collect_isaac_v3.py --minutes 15
```

Train:

```powershell
python train_decoder_v3.py
```

Run the assisted agent:

```powershell
python play_isaac_v3.py --grid-assist
```

Collect DAgger corrections:

```powershell
python dagger_isaac_v3.py
```

Then train again.

## Runtime modes

Default model selection:

```powershell
python play_isaac_v3.py
```

Grid-assisted:

```powershell
python play_isaac_v3.py --grid-assist
```

Pure visual mode:

```powershell
python play_isaac_v3.py --pure-vision
```

Pure visual mode without dopamine:

```powershell
python play_isaac_v3.py --pure-vision --no-dopamine
```

Keep dopamine neural drive but disable online action-bias adaptation:

```powershell
python play_isaac_v3.py --grid-assist --no-dopamine-plasticity
```

Keep the decoder's grid input but disable grid injection into MaleCNS:

```powershell
python play_isaac_v3.py --grid-assist --no-brain-grid
```

Disable the anatomical monitor:

```powershell
python play_isaac_v3.py --grid-assist --no-neuron-monitor
```

## Dopamine settings

The user-editable dopamine configuration is intentionally centralized in:

```text
fly_agent/isaac_v3/dopamine_settings.py
```

V3.7.1 preserves the current custom values. Do not replace this file with an
older default config when applying patches.

Persistent dopamine policy memory:

```text
fly_agent/checkpoints/isaac_dopamine_policy_v36.npz
```

This memory is separate from MaleCNS connectome weights. Current V3.7.1 does
**not** implement general persistent synaptic plasticity across the connectome.

## Door-memory anti-farming

V3.7.1 stores room connections as undirected edges:

```text
room A <-> room B
```

The first crossing can allow `NEW_ROOM` dopamine. Repeated movement through the
same connection is blocked from repeatedly generating that dopamine signal.

`ROOM_CLEAR` is also deduplicated to once per room in a run.

Files:

```text
fly_agent/isaac_v3/door_memory.py
fly_agent/isaac_v3/door_memory_settings.py
```

## Tactical projectile vision

The V3.7 projectile bridge reports hostile projectile position, velocity and
hitbox information. Python generates tactical maps for:

```text
NOW
+150 ms
+300 ms
+500 ms
```

This supports a danger corridor / approximate time-to-impact signal instead of
only exposing the nearest projectile distance.

## Performance

The current fast loop targets a faster capture stream than the MaleCNS update
rate and drops stale frames instead of allowing latency to accumulate.

Settings:

```text
fly_agent/isaac_v3/performance_settings.py
```

The important principle is **freshest frame wins**. A higher nominal FPS is not
useful if it creates a queue of old frames.

## Same-room watchdog

If the runtime remains in the same room for approximately five minutes, the
watchdog:

1. releases held controls,
2. holds `R` for three seconds,
3. restarts the run,
4. resets transient MaleCNS/retina/temporal state,
5. preserves learned decoder files and persistent dopamine policy memory.

Settings:

```text
fly_agent/isaac_v3/stuck_settings.py
```

Disable it with:

```powershell
python play_isaac_v3.py --grid-assist --no-stuck-watchdog
```

## Anatomical activity monitor

The monitor uses anatomical point positions from the MaleCNS data. It shows
live whole-brain spike activity, descending activity, retina/tactical panels,
decision probabilities and multiple anatomical projections.

It is a point-cloud anatomical activity display; it is **not** a full render of
every neuron's axon and dendrite skeleton.

## Repository structure

```text
fly.ai/
├── README.md
├── FlyIsaac_KOMENDY_V3.7.1.txt
├── requirements.txt
├── isaac_mods/
│   ├── README_MODS.md
│   ├── FlyAI_Combat_Bridge_V4/
│   └── FlyAI_V37_Projectile_Bridge/
└── fly_agent/
    ├── play_isaac_v3.py
    ├── collect_isaac_v3.py
    ├── dagger_isaac_v3.py
    ├── train_decoder_v3.py
    ├── check_isaac_bridges_v371.py
    ├── cleanup_legacy_v371.py
    ├── isaac/              # legacy project modules; do not blindly overwrite
    ├── isaac_v3/           # current FlyIsaac runtime
    ├── datasets/
    └── checkpoints/
```

## Cleanup

The current cleanup script removes a defined list of obsolete diagnostic/test
files and generated Python caches:

```powershell
python cleanup_legacy_v371.py
```

It deliberately does not delete the core source directories, datasets,
checkpoints, Isaac mods or dopamine settings.

## Data and model compatibility

V3.7 keeps the legacy 12×20 decoder grid so existing V3.5/V3.6 grid-trained
models and data do not need to be discarded solely because the new neural
tactical grid is 7×14.

Older V3.4 visual data/checkpoints can still be useful for the visual transfer
path.

## Research status / limitations

This is an experimental control system, not a validated model of fly cognition.

Important distinctions:

- the MaleCNS connectivity is biologically derived;
- pixel preprocessing is engineered;
- the tactical grid is privileged game-state information;
- grid-to-MaleCNS routing is engineered;
- PAM/PPL mapping is a computational valence mechanism built on available
  annotations;
- the current persistent online learning mechanism changes bounded action
  biases, not all MaleCNS synaptic weights;
- synthetic unit tests do not prove that every cell annotation or Lua bridge is
  available in a particular live installation.

These distinctions are kept explicit so pixel-only and assisted experiments can
be compared honestly.

## Command reference

For the full Windows command cheat-sheet, see:

```text
FlyIsaac_KOMENDY_V3.7.1.txt
```
