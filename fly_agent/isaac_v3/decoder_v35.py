from pathlib import Path
import numpy as np
from . import config


class BranchDecoderV35:
    """Separate trainable encoders for visual, grid and DN streams, then fusion."""
    VERSION = 35

    def __init__(self, mode, dn_dim, visual_dim, grid_dim,
                 context_frames=config.TEMPORAL_CONTEXT_FRAMES,
                 sample_hz=config.TEMPORAL_SAMPLE_HZ, lag_samples=0,
                 seed=config.DECODER_SEED):
        self.mode = str(mode).lower()
        if self.mode not in config.V35_INPUT_MODES:
            raise ValueError(self.mode)
        self.dn_dim, self.visual_dim, self.grid_dim = int(dn_dim), int(visual_dim), int(grid_dim)
        self.context_frames = max(1, int(context_frames))
        self.sample_hz = float(sample_hz); self.lag_samples = int(lag_samples)
        self.seed = int(seed); self.rng = np.random.default_rng(self.seed)
        self.branches = self._branches_for_mode(self.mode)
        self.branch_dims = {"visual": self.visual_dim, "grid": self.grid_dim, "dn": self.dn_dim}
        self.branch_hidden = {
            "visual": int(config.V35_VISUAL_BRANCH_HIDDEN),
            "grid": int(config.V35_GRID_BRANCH_HIDDEN),
            "dn": int(config.V35_DN_BRANCH_HIDDEN),
        }
        self.branch_features = {
            "visual": int(config.V35_VISUAL_FEATURES),
            "grid": int(config.V35_GRID_FEATURES),
            "dn": int(config.V35_DN_FEATURES),
        }
        self.base_input_dim = sum(self.branch_dims[b] for b in self.branches)
        self.full_input_dim = self.base_input_dim * self.context_frames
        self.pre = {}
        self.w_branch, self.b_branch = {}, {}
        for b in self.branches:
            raw = self.branch_dims[b] * self.context_frames
            k = min(raw, self.branch_features[b])
            self.pre[b] = dict(idx=np.arange(k, dtype=np.int64), mean=np.zeros(k, np.float32), std=np.ones(k, np.float32))
            self._init_branch(b, k)
        fused = sum(self.branch_hidden[b] for b in self.branches)
        self.shared_hidden = int(config.V35_SHARED_HIDDEN)
        s = np.sqrt(2.0/max(1,fused))
        self.w_shared = (self.rng.standard_normal((fused,self.shared_hidden))*s).astype(np.float32)
        self.b_shared = np.zeros(self.shared_hidden,np.float32)
        s2 = np.sqrt(2.0/max(1,self.shared_hidden))
        self.w_move = (self.rng.standard_normal((self.shared_hidden,len(config.MOVEMENT_CLASSES)))*s2).astype(np.float32)
        self.b_move = np.zeros(len(config.MOVEMENT_CLASSES),np.float32)
        self.w_shoot = (self.rng.standard_normal((self.shared_hidden,len(config.SHOOT_CLASSES)))*s2).astype(np.float32)
        self.b_shoot = np.zeros(len(config.SHOOT_CLASSES),np.float32)

    @staticmethod
    def _branches_for_mode(mode):
        return {
            "visual": ("visual",),
            "grid": ("grid",),
            "visual_grid": ("visual","grid"),
            "visual_grid_dn": ("visual","grid","dn"),
        }[mode]

    def _init_branch(self,b,input_dim):
        scale=np.sqrt(2.0/max(1,input_dim)); h=self.branch_hidden[b]
        self.w_branch[b]=(self.rng.standard_normal((input_dim,h))*scale).astype(np.float32)
        self.b_branch[b]=np.zeros(h,np.float32)

    @staticmethod
    def _softmax(x):
        x=np.asarray(x,np.float32); y=x-np.max(x,axis=-1,keepdims=True); e=np.exp(np.clip(y,-50,50)); return e/np.maximum(e.sum(axis=-1,keepdims=True),1e-8)

    def _slice_map(self):
        pos=0; out={}
        for b in self.branches:
            d=self.branch_dims[b]; out[b]=(pos,pos+d); pos+=d
        return out

    def _branch_raw(self,x,b):
        x=np.asarray(x,np.float32); one=x.ndim==1
        if one:x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=self.full_input_dim:
            raise ValueError(f"V3.5 decoder expects {self.full_input_dim}, got {x.shape}")
        frames=x.reshape(len(x),self.context_frames,self.base_input_dim)
        a,z=self._slice_map()[b]
        raw=frames[:,:,a:z].reshape(len(x),-1)
        return raw,one

    def fit_preprocessor(self, x, preserve_branches=()):
        preserve = set(str(b) for b in preserve_branches)
        for b in self.branches:
            raw, _ = self._branch_raw(x, b)
            if b in preserve:
                p = self.pre.get(b)
                if p is None or len(p.get("idx", [])) == 0 or int(np.max(p["idx"])) >= raw.shape[1]:
                    raise ValueError(f"Cannot preserve incompatible {b} preprocessor")
                continue
            var = np.var(raw, axis=0)
            k = min(self.branch_features[b], raw.shape[1])
            if k < raw.shape[1]:
                idx = np.argpartition(var, -k)[-k:]
                idx = idx[np.argsort(var[idx])[::-1]]
            else:
                idx = np.arange(raw.shape[1])
            sel = raw[:, idx]
            mean = sel.mean(0).astype(np.float32)
            std = sel.std(0).astype(np.float32)
            std[std < 1e-5] = 1.0
            self.pre[b] = dict(idx=idx.astype(np.int64), mean=mean, std=std)
            self._init_branch(b, len(idx))

    def copy_branch_from(self, other, branch="visual"):
        """Copy one compatible learned branch from another V3.5 decoder."""
        branch = str(branch)
        if branch not in self.branches or branch not in other.branches:
            raise ValueError(f"Branch {branch!r} must exist in both decoders")
        if self.branch_dims[branch] != other.branch_dims[branch]:
            raise ValueError(f"Branch dimension mismatch for {branch}")
        if self.branch_hidden[branch] != other.branch_hidden[branch]:
            raise ValueError(f"Branch hidden mismatch for {branch}")
        p = other.pre[branch]
        self.pre[branch] = dict(
            idx=np.asarray(p["idx"], dtype=np.int64).copy(),
            mean=np.asarray(p["mean"], dtype=np.float32).copy(),
            std=np.asarray(p["std"], dtype=np.float32).copy(),
        )
        self.w_branch[branch] = np.asarray(other.w_branch[branch], dtype=np.float32).copy()
        self.b_branch[branch] = np.asarray(other.b_branch[branch], dtype=np.float32).copy()
        return self

    def warm_start_from_visual(self, other):
        """Seed an assisted model from an already learned V3.5 visual model.

        The visual encoder, shared hidden bias, action heads, and the visual rows
        of the fusion matrix are copied. New GRID/DN rows remain freshly
        initialized and can learn without erasing the old visual policy.
        """
        if other.mode != "visual" or "visual" not in self.branches:
            raise ValueError("warm_start_from_visual requires a visual teacher and visual student branch")
        self.copy_branch_from(other, "visual")
        vh = self.branch_hidden["visual"]
        if other.w_shared.shape[0] != vh or self.w_shared.shape[1] != other.w_shared.shape[1]:
            raise ValueError("Shared layer shape mismatch")
        self.w_shared[:vh, :] = np.asarray(other.w_shared, dtype=np.float32)
        self.b_shared = np.asarray(other.b_shared, dtype=np.float32).copy()
        self.w_move = np.asarray(other.w_move, dtype=np.float32).copy()
        self.b_move = np.asarray(other.b_move, dtype=np.float32).copy()
        self.w_shoot = np.asarray(other.w_shoot, dtype=np.float32).copy()
        self.b_shoot = np.asarray(other.b_shoot, dtype=np.float32).copy()
        self.transfer_teacher = getattr(other, "transfer_teacher", "") or "V3.5 visual branch"
        return self

    def _prepared(self,x):
        out={}; one=None
        for b in self.branches:
            raw,o=self._branch_raw(x,b); one=o if one is None else one; p=self.pre[b]
            out[b]=(raw[:,p['idx']]-p['mean'])/p['std']
        return out,one

    def _forward(self,z):
        hb={b:np.tanh(z[b]@self.w_branch[b]+self.b_branch[b]) for b in self.branches}
        fused=np.concatenate([hb[b] for b in self.branches],axis=1)
        h=np.tanh(fused@self.w_shared+self.b_shared)
        return hb,fused,h,self._softmax(h@self.w_move+self.b_move),self._softmax(h@self.w_shoot+self.b_shoot)

    def predict_proba(self,x):
        z,one=self._prepared(x); *_,mp,sp=self._forward(z); return (mp[0],sp[0]) if one else (mp,sp)

    def predict_actions(self,x,threshold=None):
        mp,sp=self.predict_proba(x)
        if mp.ndim!=1: raise ValueError("predict_actions expects one temporal vector")
        mi,si=int(np.argmax(mp)),int(np.argmax(sp)); mn=config.MOVEMENT_CLASSES[mi]; sn=config.SHOOT_CLASSES[si]
        actions=set(config.MOVEMENT_ACTIONS[mn]); actions.update(config.SHOOT_ACTIONS[sn])
        return actions,{"move_index":mi,"shoot_index":si,"move_name":mn,"shoot_name":sn,"move_confidence":float(mp[mi]),"shoot_confidence":float(sp[si]),"move_probs":mp,"shoot_probs":sp}

    @staticmethod
    def _class_weights(t,n):
        c=np.maximum(np.bincount(np.asarray(t,np.int64),minlength=n).astype(np.float32),1.0); w=np.sqrt(len(t)/c); w/=max(1e-6,float(w.mean())); return np.clip(w,0.5,3.0)
    @staticmethod
    def _ce(t,p,cw=None,sw=None):
        t=np.asarray(t,np.int64); p=np.asarray(p,np.float32); q=np.clip(p[np.arange(len(t)),t],1e-7,1); w=np.ones(len(t),np.float32)
        if cw is not None:w*=cw[t]
        if sw is not None:w*=np.asarray(sw,np.float32)
        return float(np.sum(-np.log(q)*w)/max(1e-8,float(w.sum())))

    def _params(self):
        params=[]; names=[]
        for b in self.branches:
            names += [("wb",b),("bb",b)]; params += [self.w_branch[b],self.b_branch[b]]
        names += [("ws",None),("bs",None),("wm",None),("bm",None),("wsh",None),("bsh",None)]
        params += [self.w_shared,self.b_shared,self.w_move,self.b_move,self.w_shoot,self.b_shoot]
        return names,params
    def _snapshot(self): return [p.copy() for p in self._params()[1]]
    def _restore(self,snap):
        names,params=self._params()
        for p,s in zip(params,snap): p[...] = s

    def train(self,x,move,shoot,sample_weights=None,epochs=30,batch_size=256,learning_rate=1e-3,validation=None,patience=config.EARLY_STOP_PATIENCE,min_delta=config.EARLY_STOP_MIN_DELTA,verbose=True,fit_preprocessor=True,preserve_branches=()):
        x=np.asarray(x,np.float32); move=np.asarray(move,np.int64); shoot=np.asarray(shoot,np.int64)
        sw=np.ones(len(x),np.float32) if sample_weights is None else np.asarray(sample_weights,np.float32)
        if fit_preprocessor:
            self.fit_preprocessor(x, preserve_branches=preserve_branches)
        z,_=self._prepared(x)
        mcw=self._class_weights(move,len(config.MOVEMENT_CLASSES)); scw=self._class_weights(shoot,len(config.SHOOT_CLASSES))
        names,params=self._params(); m=[np.zeros_like(p) for p in params]; v=[np.zeros_like(p) for p in params]
        beta1,beta2,eps=.9,.999,1e-8; step=0; best=np.inf; best_epoch=0; best_snap=None; bad=0; hist=[]
        for epoch in range(1,int(epochs)+1):
            order=self.rng.permutation(len(x)); losses=[]
            for st in range(0,len(x),int(batch_size)):
                idx=order[st:st+int(batch_size)]; zb={b:z[b][idx] for b in self.branches}; mb=move[idx]; sb=shoot[idx]; wb=sw[idx]
                hb,fused,h,mp,sp=self._forward(zb); losses.append(self._ce(mb,mp,mcw,wb)+self._ce(sb,sp,scw,wb))
                dm=mp.copy(); dm[np.arange(len(idx)),mb]-=1; wm=wb*mcw[mb]; dm*= (wm/max(1e-8,float(wm.sum())))[:,None]
                ds=sp.copy(); ds[np.arange(len(idx)),sb]-=1; wsamp=wb*scw[sb]; ds*= (wsamp/max(1e-8,float(wsamp.sum())))[:,None]
                gwm=h.T@dm; gbm=dm.sum(0); gwsh=h.T@ds; gbsh=ds.sum(0)
                dh=(dm@self.w_move.T+ds@self.w_shoot.T)*(1-h*h); gws=fused.T@dh; gbs=dh.sum(0)
                df=(dh@self.w_shared.T); grads=[]; pos=0
                for b in self.branches:
                    width=self.branch_hidden[b]; d= df[:,pos:pos+width]*(1-hb[b]*hb[b]); pos+=width
                    grads += [zb[b].T@d,d.sum(0)]
                grads += [gws,gbs,gwm,gbm,gwsh,gbsh]
                step+=1
                for i,(p,g) in enumerate(zip(params,grads)):
                    m[i]=beta1*m[i]+(1-beta1)*g; v[i]=beta2*v[i]+(1-beta2)*(g*g); mh=m[i]/(1-beta1**step); vh=v[i]/(1-beta2**step); p-=float(learning_rate)*mh/(np.sqrt(vh)+eps)
            train_loss=float(np.mean(losses)); rec={"epoch":epoch,"train_loss":train_loss}
            if validation is not None:
                vx,vm,vs=validation; vmp,vsp=self.predict_proba(vx); vl=self._ce(vm,vmp)+self._ce(vs,vsp); mhat=np.argmax(vmp,1); shat=np.argmax(vsp,1); ma=float((mhat==vm).mean()); sa=float((shat==vs).mean()); ja=float(((mhat==vm)&(shat==vs)).mean())
                rec.update(val_loss=vl,move_accuracy=ma,shoot_accuracy=sa,joint_accuracy=ja)
                if vl<best-float(min_delta): best=vl; best_epoch=epoch; best_snap=self._snapshot(); bad=0
                else: bad+=1
            hist.append(rec)
            if verbose:
                txt=f"epoch {epoch:02d} | train={train_loss:.4f}"
                if validation is not None: txt+=f" | val={rec['val_loss']:.4f} | move={rec['move_accuracy']:.3f} | shoot={rec['shoot_accuracy']:.3f} | joint={rec['joint_accuracy']:.3f} | best={best_epoch:02d}"
                print(txt)
            if validation is not None and bad>=int(patience):
                if verbose: print(f"EARLY STOP at epoch {epoch}; restoring best epoch {best_epoch} (val={best:.4f})")
                break
        if best_snap is not None:self._restore(best_snap)
        self.best_epoch=int(best_epoch if best_epoch else hist[-1]['epoch']); return hist

    def distill_soft(self, x, teacher_move, teacher_shoot, epochs=8, batch_size=256, learning_rate=7e-4, verbose=True):
        """Pretrain this decoder to imitate soft probabilities from an older teacher."""
        x = np.asarray(x, dtype=np.float32)
        tm = np.asarray(teacher_move, dtype=np.float32)
        ts = np.asarray(teacher_shoot, dtype=np.float32)
        if len(x) != len(tm) or len(x) != len(ts):
            raise ValueError("Distillation sample count mismatch")
        if tm.shape[1] != len(config.MOVEMENT_CLASSES) or ts.shape[1] != len(config.SHOOT_CLASSES):
            raise ValueError("Teacher probability width mismatch")
        self.fit_preprocessor(x)
        z, _ = self._prepared(x)
        _names, params = self._params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = .9, .999, 1e-8
        step = 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            order = self.rng.permutation(len(x))
            losses = []
            for st in range(0, len(x), int(batch_size)):
                idx = order[st:st + int(batch_size)]
                zb = {b: z[b][idx] for b in self.branches}
                qmove = tm[idx]
                qshoot = ts[idx]
                hb, fused, h, mp, sp = self._forward(zb)
                loss = float(
                    np.mean(-np.sum(qmove * np.log(np.clip(mp, 1e-7, 1.0)), axis=1))
                    + np.mean(-np.sum(qshoot * np.log(np.clip(sp, 1e-7, 1.0)), axis=1))
                )
                losses.append(loss)
                n = max(1, len(idx))
                dm = (mp - qmove) / float(n)
                ds = (sp - qshoot) / float(n)
                gwm = h.T @ dm; gbm = dm.sum(0)
                gwsh = h.T @ ds; gbsh = ds.sum(0)
                dh = (dm @ self.w_move.T + ds @ self.w_shoot.T) * (1 - h*h)
                gws = fused.T @ dh; gbs = dh.sum(0)
                df = dh @ self.w_shared.T
                grads = []
                pos = 0
                for b in self.branches:
                    width = self.branch_hidden[b]
                    d = df[:, pos:pos+width] * (1 - hb[b]*hb[b])
                    pos += width
                    grads += [zb[b].T @ d, d.sum(0)]
                grads += [gws, gbs, gwm, gbm, gwsh, gbsh]
                step += 1
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = beta1*m[i] + (1-beta1)*g
                    v[i] = beta2*v[i] + (1-beta2)*(g*g)
                    mh = m[i] / (1-beta1**step)
                    vh = v[i] / (1-beta2**step)
                    p -= float(learning_rate) * mh / (np.sqrt(vh) + eps)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            history.append(mean_loss)
            if verbose:
                print(f"distill {epoch:02d} | soft_ce={mean_loss:.4f}")
        self.transfer_teacher = getattr(self, "transfer_teacher", "V3.4")
        self.transfer_distill_epochs = int(epochs)
        return history

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        d=dict(version=np.asarray([35],np.int32),mode=np.asarray([self.mode]),dn_dim=np.asarray([self.dn_dim],np.int32),visual_dim=np.asarray([self.visual_dim],np.int32),grid_dim=np.asarray([self.grid_dim],np.int32),context_frames=np.asarray([self.context_frames],np.int32),sample_hz=np.asarray([self.sample_hz],np.float32),lag_samples=np.asarray([self.lag_samples],np.int32),w_shared=self.w_shared,b_shared=self.b_shared,w_move=self.w_move,b_move=self.b_move,w_shoot=self.w_shoot,b_shoot=self.b_shoot,best_epoch=np.asarray([getattr(self,'best_epoch',0)],np.int32),transfer_teacher=np.asarray([getattr(self,'transfer_teacher','')]),transfer_distill_epochs=np.asarray([getattr(self,'transfer_distill_epochs',0)],np.int32))
        for b in self.branches:
            p=self.pre[b]; d[f'{b}_idx']=p['idx']; d[f'{b}_mean']=p['mean']; d[f'{b}_std']=p['std']; d[f'{b}_w']=self.w_branch[b]; d[f'{b}_b']=self.b_branch[b]
        np.savez_compressed(path,**d); print("Decoder V3.5 saved:",path)

    @classmethod
    def load(cls,path):
        data=np.load(path,allow_pickle=False); version=int(data['version'][0]) if 'version' in data else 0
        if version!=35: raise RuntimeError(f"Decoder {path} is format v{version}; V3.5 needs v35")
        obj=cls(mode=str(data['mode'][0]),dn_dim=int(data['dn_dim'][0]),visual_dim=int(data['visual_dim'][0]),grid_dim=int(data['grid_dim'][0]),context_frames=int(data['context_frames'][0]),sample_hz=float(data['sample_hz'][0]),lag_samples=int(data['lag_samples'][0]))
        for b in obj.branches:
            obj.pre[b]=dict(idx=data[f'{b}_idx'].astype(np.int64),mean=data[f'{b}_mean'].astype(np.float32),std=data[f'{b}_std'].astype(np.float32)); obj.w_branch[b]=data[f'{b}_w'].astype(np.float32); obj.b_branch[b]=data[f'{b}_b'].astype(np.float32)
        obj.w_shared=data['w_shared'].astype(np.float32); obj.b_shared=data['b_shared'].astype(np.float32); obj.w_move=data['w_move'].astype(np.float32); obj.b_move=data['b_move'].astype(np.float32); obj.w_shoot=data['w_shoot'].astype(np.float32); obj.b_shoot=data['b_shoot'].astype(np.float32); obj.best_epoch=int(data['best_epoch'][0]) if 'best_epoch' in data else 0
        obj.transfer_teacher=str(data['transfer_teacher'][0]) if 'transfer_teacher' in data else ''
        obj.transfer_distill_epochs=int(data['transfer_distill_epochs'][0]) if 'transfer_distill_epochs' in data else 0
        return obj
