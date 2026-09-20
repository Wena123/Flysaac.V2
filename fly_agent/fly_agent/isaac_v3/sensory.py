import numpy as np
from . import config


class HybridVisualEncoder:
    """Retina + optional V3.7 tactical grid -> real MaleCNS sensory neurons."""
    def __init__(self, fly_agent_brain):
        self.fly_agent_brain=fly_agent_brain; self.brain=fly_agent_brain.brain
        self.lc10_l=self._cells(["LC10a"],side="L"); self.lc10_r=self._cells(["LC10a"],side="R")
        self.lplc1_l=self._cells(["LPLC1"],side="L",optional=True); self.lplc1_r=self._cells(["LPLC1"],side="R",optional=True)
        self.lplc2_l=self._cells(["LPLC2"],side="L",optional=True); self.lplc2_r=self._cells(["LPLC2"],side="R",optional=True)
        self.lc4_l=self._cells(["LC4"],side="L",optional=True); self.lc4_r=self._cells(["LC4"],side="R",optional=True)
        if self.lc10_l.size==0 or self.lc10_r.size==0: raise RuntimeError("FlyIsaac requires LC10a cells on both sides.")
        ntiles=config.LC10_TILES_Y*config.LC10_TILES_X_PER_SIDE
        self._lc10_groups_l=[g for g in np.array_split(self.lc10_l,ntiles) if len(g)]
        self._lc10_groups_r=[g for g in np.array_split(self.lc10_r,ntiles) if len(g)]
        self.last_grid_field=None; self.last_grid_injection_count=0; self.last_tactical=None
        print("FlyIsaac V3.7 visual neurons:")
        print("  LC10a L/R:",len(self.lc10_l),"/",len(self.lc10_r)); print("  LPLC1 L/R:",len(self.lplc1_l),"/",len(self.lplc1_r)); print("  LPLC2 L/R:",len(self.lplc2_l),"/",len(self.lplc2_r)); print("  LC4   L/R:",len(self.lc4_l),"/",len(self.lc4_r))
    def _cells(self,names,side=None,optional=False):
        try: return np.asarray(self.brain.cells(names) if side is None else self.brain.cells(names,side=side),dtype=np.int64)
        except Exception as exc:
            if optional: print(f"  warning: no {names} side={side}: {exc}"); return np.empty(0,np.int64)
            raise
    @staticmethod
    def _tile_activity(field,tiles_y,tiles_x):
        field=np.asarray(field,np.float32); ys=np.linspace(0,field.shape[0],tiles_y+1,dtype=int); xs=np.linspace(0,field.shape[1],tiles_x+1,dtype=int); out=[]
        for iy in range(tiles_y):
            for ix in range(tiles_x):
                t=field[ys[iy]:ys[iy+1],xs[ix]:xs[ix+1]]; out.append(0.0 if not t.size else .35*float(t.mean())+.65*float(t.max()))
        return np.asarray(out,np.float32)
    @staticmethod
    def _append_scalar(inject,cells,amount):
        if cells is None or len(cells)==0 or float(amount)<=0: return
        inject.append((np.asarray(cells,np.int64),float(amount)))
    def _append_lc10_side(self,inject,field,groups,gain=None,max_drive=None,minimum=None,top_k=None):
        values=self._tile_activity(field,config.LC10_TILES_Y,config.LC10_TILES_X_PER_SIDE)
        gain=config.LC10_GAIN if gain is None else float(gain); max_drive=config.LC10_MAX_DRIVE if max_drive is None else float(max_drive); minimum=config.LC10_MIN_ACTIVITY if minimum is None else float(minimum); top_k=config.LC10_TOP_K_TILES_PER_SIDE if top_k is None else int(top_k)
        active=np.flatnonzero(values>=minimum)
        if not active.size: return 0
        order=active[np.argsort(values[active])[-min(top_k,active.size):]]; count=0
        for tile in order:
            if tile>=len(groups): continue
            before=len(inject); self._append_scalar(inject,groups[tile],min(max_drive,gain*float(values[tile]))); count+=int(len(inject)>before)
        return count
    @staticmethod
    def _resample(field,rows,cols):
        a=np.asarray(field,np.float32)
        if a.ndim!=2 or not a.size: return np.zeros((rows,cols),np.float32)
        rr=np.clip(np.round((np.arange(rows)+.5)*a.shape[0]/rows-.5).astype(int),0,a.shape[0]-1); cc=np.clip(np.round((np.arange(cols)+.5)*a.shape[1]/cols-.5).astype(int),0,a.shape[1]-1)
        return np.clip(np.nan_to_num(a[rr][:,cc]),0,1).astype(np.float32)
    def _legacy_grid_field(self,features):
        if not features: return None
        field=np.zeros((12,20),np.float32)
        for name,weight in config.V36_GRID_BRAIN_WEIGHTS.items():
            a=self._resample(features.get(name,np.zeros((1,1),np.float32)),12,20)
            field=np.maximum(field,np.float32(weight)*a)
        return np.clip(field.reshape(6,2,20).max(axis=1),0,1)
    def _append_assisted(self,inject,grid_features=None,tactical=None):
        self.last_tactical=tactical
        if tactical is not None and tactical.get("neural_field") is not None:
            source=np.asarray(tactical["neural_field"],np.float32)
            field=self._resample(source,6,20)
            gain=config.V37_TACTICAL_LC10_GAIN; max_drive=config.V37_TACTICAL_LC10_MAX_DRIVE; minimum=config.V37_TACTICAL_LC10_MIN_ACTIVITY; top_k=config.V37_TACTICAL_LC10_TOP_K_PER_SIDE
        else:
            field=self._legacy_grid_field(grid_features)
            gain=config.V36_GRID_BRAIN_GAIN; max_drive=config.V36_GRID_BRAIN_MAX_DRIVE; minimum=config.V36_GRID_BRAIN_MIN_ACTIVITY; top_k=config.V36_GRID_BRAIN_TOP_K_PER_SIDE
        self.last_grid_field=field; self.last_grid_injection_count=0
        if field is None: return
        mid=field.shape[1]//2; kw=dict(gain=gain,max_drive=max_drive,minimum=minimum,top_k=top_k)
        n=self._append_lc10_side(inject,field[:,:mid],self._lc10_groups_l,**kw)+self._append_lc10_side(inject,field[:,mid:],self._lc10_groups_r,**kw); self.last_grid_injection_count=int(n)
        if tactical is not None:
            dl=float(tactical.get("danger_left",0.0)); dr=float(tactical.get("danger_right",0.0)); cap=float(config.V37_PROJECTILE_MAX_EXTRA_DRIVE)
            self._append_scalar(inject,self.lplc2_l,min(cap,config.V37_PROJECTILE_LPLC2_GAIN*dl)); self._append_scalar(inject,self.lplc2_r,min(cap,config.V37_PROJECTILE_LPLC2_GAIN*dr))
            self._append_scalar(inject,self.lc4_l,min(cap,config.V37_PROJECTILE_LC4_GAIN*dl)); self._append_scalar(inject,self.lc4_r,min(cap,config.V37_PROJECTILE_LC4_GAIN*dr))
    def make_injections(self,retina,grid_features=None,tactical=None,dopamine=None):
        inject=[]; mid=retina.target.shape[1]//2
        red=np.asarray(retina.red,np.float32); green=np.asarray(retina.green,np.float32); blue=np.asarray(retina.blue,np.float32); chroma=np.maximum(np.abs(red-green),np.abs(blue-.5*(red+green)))
        sal=np.maximum.reduce([np.asarray(retina.target,np.float32),.58*np.asarray(retina.motion,np.float32),.42*np.asarray(retina.motion,np.float32)*np.asarray(retina.edge,np.float32),.24*np.asarray(retina.motion,np.float32)*np.asarray(retina.contrast,np.float32),.12*np.asarray(retina.motion,np.float32)*chroma]); sal=np.clip(sal,0,1)
        self._append_lc10_side(inject,sal[:,:mid],self._lc10_groups_l); self._append_lc10_side(inject,sal[:,mid:],self._lc10_groups_r)
        ml=np.maximum(np.asarray(retina.motion[:,:mid],np.float32),np.maximum(np.asarray(retina.motion_left[:,:mid],np.float32),np.asarray(retina.motion_right[:,:mid],np.float32))); mr=np.maximum(np.asarray(retina.motion[:,mid:],np.float32),np.maximum(np.asarray(retina.motion_left[:,mid:],np.float32),np.asarray(retina.motion_right[:,mid:],np.float32)))
        self._append_scalar(inject,self.lplc1_l,config.LPLC1_GAIN*float(np.percentile(ml,95))); self._append_scalar(inject,self.lplc1_r,config.LPLC1_GAIN*float(np.percentile(mr,95)))
        self._append_scalar(inject,self.lplc2_l,config.LPLC2_GAIN*retina.looming_left); self._append_scalar(inject,self.lplc2_r,config.LPLC2_GAIN*retina.looming_right); self._append_scalar(inject,self.lc4_l,config.LC4_GAIN*retina.looming_left); self._append_scalar(inject,self.lc4_r,config.LC4_GAIN*retina.looming_right)
        self._append_assisted(inject,grid_features=grid_features,tactical=tactical)
        if dopamine is not None:
            try: inject.extend(dopamine.make_injections())
            except Exception as exc:
                if not getattr(self,"_dopamine_warned",False): print("Dopamine injection warning:",exc); self._dopamine_warned=True
        return inject
    def step(self,retina,grid_features=None,tactical=None,dopamine=None):
        inject=self.make_injections(retina,grid_features=grid_features,tactical=tactical,dopamine=dopamine); return self.brain.step(inject=inject) if inject else self.brain.step()
