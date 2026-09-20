import numpy as np
from isaac_v3.sensory import HybridVisualEncoder

class Core:
    def __init__(self):
        self.last_inject=[]
    def cells(self,names,side=None):
        name=names[0]
        base={"LC10a":0,"LPLC1":300,"LPLC2":400,"LC4":500}[name]
        n={"LC10a":140,"LPLC1":70,"LPLC2":90,"LC4":60}[name]
        if side=="R": base+=700
        return np.arange(base,base+n,dtype=np.int64)
    def step(self,inject=None):
        self.last_inject=list(inject or [])
        return np.array([1,2,3],dtype=np.int64)
class Fly:
    def __init__(self): self.brain=Core()
class Retina:
    def __init__(self):
        z=np.zeros((60,60),np.float32)
        self.target=z.copy(); self.motion=z.copy(); self.edge=z.copy(); self.contrast=z.copy(); self.red=z.copy(); self.green=z.copy(); self.blue=z.copy(); self.motion_left=z.copy(); self.motion_right=z.copy(); self.looming_left=0.; self.looming_right=0.

fly=Fly(); enc=HybridVisualEncoder(fly)
tg=np.zeros((7,14),np.float32); tg[2:5,1:4]=1.; tg[2:5,10:13]=1.
tactical={"neural_field":tg,"danger_left":1.0,"danger_right":.7}
enc.step(Retina(), tactical=tactical)
assert enc.last_grid_injection_count>0
# Tactical projectile danger should create injections beyond LC10 (LPLC2/LC4 scalar sets)
sets=[set(np.asarray(cells).tolist()) for cells,amount in fly.brain.last_inject]
assert any(400 in s for s in sets), 'left LPLC2 missing'
assert any(500 in s for s in sets), 'left LC4 missing'
print('SENSORY V3.7 TEST: OK')
print('LC10 tactical injections:',enc.last_grid_injection_count,'total injection groups:',len(fly.brain.last_inject))
