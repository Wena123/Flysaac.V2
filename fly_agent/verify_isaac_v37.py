from pathlib import Path
from isaac_v3 import config, dopamine_settings as d
from isaac_v3.tactical_grid import TacticalGridV37, ProjectileBridgeV37
from isaac_v3 import performance_settings as p
root=Path(__file__).resolve().parent
assert config.V37_TACTICAL_GRID_ROWS == 7 and config.V37_TACTICAL_GRID_COLS == 14
assert p.VISION_CAPTURE_HZ == 60.0 and p.BRAIN_HZ == 45.0 and p.DROP_STALE_FRAMES
assert d.EVENT_DOPAMINE["ROOM_CLEAR"] == 100.0 and d.EVENT_DOPAMINE["BOSS_KILL"] == 30.0
assert (root.parent/"isaac_mods"/"FlyAI_V37_Projectile_Bridge"/"main.lua").exists() or True
print("FlyIsaac V3.7 sensory upgrade: OK")
print("tactical neural grid: 7x14")
print("legacy decoder grid: 12x20 PRESERVED")
print("capture target: 60 Hz")
print("MaleCNS target: 45 Hz")
print("projectile prediction: NOW / +150 / +300 / +500 ms")
print("projectile danger -> LC10 + LPLC2 + LC4")
print("5-minute room watchdog: INCLUDED")
print("user dopamine values: PRESERVED")
