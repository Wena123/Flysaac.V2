import numpy as np
from isaac_v3.neuron_monitor import NeuronMonitor
class B:
    def __init__(self):
        rng=np.random.default_rng(3); self.positions=rng.normal(size=(5000,3)).astype(np.float32); self.cell_type=np.array(["LC10a"]*100+["LPLC2"]*50+["LC4"]*50+["PAM01"]*20+["PPL01"]*20+[""]*4760); self.superclass=np.array([""]*5000)
    def cells(self,names,side=None):
        if "descending_neuron" in names: return np.arange(1000,1314)
        return np.flatnonzero(np.isin(self.cell_type,names))
class R:
    target=np.zeros((60,60),np.float32); motion=np.zeros((60,60),np.float32); looming_left=.2; looming_right=.3
b=B(); m=NeuronMonitor(brain=b,dn_cells=np.arange(1000,1314),enabled=False)
tg=np.zeros((7,14),np.float32); tg[3,4]=1
ch={k:tg.copy() for k in ["projectile_now_grid","projectile_future_150_grid","projectile_future_300_grid","projectile_future_500_grid"]}
canvas=m.render(np.zeros(314,np.float32),fired=np.array([1,101,155,205,225,1005]),retina=R(),tactical={"neural_field":tg,"channels":ch,"projectile_count":2,"min_tti":.22,"projectile_bridge_ok":True},perf={"capture_fps":58.2,"brain_fps":44.6,"dropped_frames":12})
assert canvas.shape == (1050,1800,3), canvas.shape
print("ANATOMY V3.7 TEST: OK", canvas.shape)
