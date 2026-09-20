import numpy as np
from isaac_v3.tactical_grid import TacticalGridV37, ProjectileBridgeV37

p = ProjectileBridgeV37.parse_state_payload("84|0.500|0.500|0.020|0.030|1|0.20,0.50,0.80,0.00,0.010,0.015")
assert p and len(p["projectiles"]) == 1
legacy = {"enemy_grid": np.zeros((12,20),np.float32), "collision_grid": np.zeros((12,20),np.float32), "self_grid": np.zeros((12,20),np.float32)}
t = TacticalGridV37().build(legacy, dict(p, has_state=True, age=0.01))
assert t["neural_field"].shape == (7,14)
assert t["channels"]["projectile_now_grid"].shape == (7,14)
now = t["channels"]["projectile_now_grid"]
f500 = t["channels"]["projectile_future_500_grid"]
assert not np.allclose(now, f500)
assert t["projectile_count"] == 1
assert t["min_tti"] is not None and 0 <= t["min_tti"] <= .8
print("TACTICAL GRID V3.7 TEST: OK")
print("grid:", t["neural_field"].shape, "projectiles:", t["projectile_count"], "tti_ms:", round(t["min_tti"]*1000))
