# Changelog

## V3.7.1

### Door-memory anti-farming
- remembers traversed room connections as undirected A↔B edges
- repeated crossings of the same connection do not repeatedly receive `NEW_ROOM` dopamine credit
- de-duplicates `ROOM_CLEAR` reward per room/run

### Repository
- polished public README
- added GitHub Actions Python syntax check
- added repository `.gitignore`
- added contribution notes

## V3.7

### Sensory / tactical grid
- added 7×14 tactical neural grid for MaleCNS sensory input
- kept legacy 12×20 decoder grid for compatibility
- added detailed hostile projectile position, velocity and hitbox stream
- added NOW / +150 ms / +300 ms / +500 ms projectile prediction
- added engineered projectile-danger sensory routing
- added faster capture / brain loop separation and stale-frame dropping
- expanded anatomical activity monitor

### Runtime
- added same-room watchdog
- after ~5 minutes in the same room: release controls, hold R for 3 seconds, restart, reset transient runtime state

## V3.6.2
- centralized dopamine tuning in `isaac_v3/dopamine_settings.py`
- preserved custom event-specific dopamine values

## V3.6
- added PAM/PPL-like dopamine signalling
- added bounded dopamine-gated online action-bias memory
- added grid-to-MaleCNS assisted sensory path

## Earlier versions
V3.1–V3.5 introduced the temporal decoder, richer retina features, multimodal decoder modes, anatomical monitoring, grid assist and V3.4→V3.5 transfer support.
