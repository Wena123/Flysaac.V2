# Isaac mods used by FlyIsaac V3.7.1

This folder bundles the **source-confirmed bridge mods available in the current
FlyIsaac release artifacts**:

1. `FlyAI_Combat_Bridge_V4`
   - emits `FLYCOMBAT|`
   - accurate player/enemy hitboxes
   - enemy state
   - legacy hostile projectile summary
   - SHOT_HIT / SHOT_MISS events

2. `FlyAI_V37_Projectile_Bridge`
   - emits `FLYCOMBATV37|`
   - detailed hostile projectile x/y, velocity and hitbox data
   - used by V3.7 tactical projectile prediction

## Important: legacy core bridge streams

The Python project also consumes the older project streams:

- `FLYAI|` — reward / run events
- `FLYROOM|` — room geometry, room index, previous room, doors
- `FLYWORLD|` — optional pickup/item/hazard/trapdoor state

Those legacy bridge source folders were **not present in the V3.7/V3.7.1
release artifacts or file library used to build this package**, so I have not
invented replacements and labelled them as originals.

Keep your already-working legacy FlyAI / room / world bridge mods installed.
The V3.7 projectile bridge runs alongside them.

Run:

    python check_isaac_bridges_v371.py

while Isaac is running to see which prefixes are actually reaching `log.txt`.

If you later add the original legacy mod folders to the repository, place them
under `isaac_mods/` too so the GitHub checkout contains the complete game-side
bridge suite.
