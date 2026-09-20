from pathlib import Path
from isaac_v3 import door_memory_settings as dm

root=Path(__file__).resolve().parent
assert (root/'isaac_v3'/'door_memory.py').exists()
assert 'dopamine.observe_room(grid_features)' in (root/'play_isaac_v3.py').read_text(encoding='utf-8')
assert 'dopamine.observe_room(grid_features)' in (root/'collect_isaac_v3.py').read_text(encoding='utf-8')
assert 'dopamine.observe_room(grid_features)' in (root/'dagger_isaac_v3.py').read_text(encoding='utf-8')
assert dm.GATE_NEW_ROOM_BY_DOOR is True
assert dm.GATE_ROOM_CLEAR_ONCE_PER_ROOM is True
print('FlyIsaac V3.7.1 door-memory patch: OK')
print('dopamine_settings.py is not included in this patch')
print('NEW_ROOM: first A<->B only')
print('ROOM_CLEAR: once per room')
