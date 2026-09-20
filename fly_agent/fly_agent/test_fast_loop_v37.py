import sys, types, time
sys.modules.setdefault('pydirectinput', types.SimpleNamespace(keyDown=lambda x:None,keyUp=lambda x:None,click=lambda *a,**k:None))
sys.modules.setdefault('win32gui', types.SimpleNamespace(GetForegroundWindow=lambda:0))
from isaac_v3.runtime import FastVisionLoop

class Stream:
    def __init__(self): self.seq=0
    def consume(self,after_sequence,timeout):
        time.sleep(.003)
        self.seq+=1
        return object(),None,self.seq,1,{}

loop=FastVisionLoop(Stream(),brain_hz=45,drop_stale=True)
for _ in range(12): loop.next_frame(timeout=.2)
p=loop.perf_snapshot()
assert p['brain_fps']>0 and p['capture_fps']>=p['brain_fps']
print('FAST LOOP V3.7 TEST: OK')
print('capture_fps=%.1f brain_fps=%.1f dropped=%d'%(p['capture_fps'],p['brain_fps'],p['dropped_frames']))
