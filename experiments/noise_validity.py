"""Numerical audit of the historical noisy-overlap similarity (not a PSD proof)."""
import json
from pathlib import Path
import numpy as np
import torch
from quantum.feature_maps import FEATURE_MAPS
from quantum.noise import build_noisy_kernel_bank
from experiments.expert_imitation import dump


def run():
    torch.set_num_threads(1)
    r=json.loads(Path('results/continuation/wine/seed_000.json').read_text())
    x=np.array(r['X_train'])[:12];theta=torch.tensor(r['theta'],dtype=torch.float64)
    out=[]
    for channel in ['depolarizing','bit_flip','phase_flip']:
        for level in [0.,.05,.1]:
            bank=build_noisy_kernel_bank(FEATURE_MAPS,4,channel,level)
            for name,k in bank.items():
                with torch.no_grad():
                    a=k.matrix(x,x,theta=theta if k.trainable else None,symmetric=False)
                    a=a.numpy() if hasattr(a,'numpy') else a
                    mirrored=k.matrix(x,x,theta=theta if k.trainable else None,symmetric=True)
                    mirrored=mirrored.numpy() if hasattr(mirrored,'numpy') else mirrored
                out.append({'channel':channel,'level':level,'kernel':name,
                            'max_asymmetry_unmirrored':float(abs(a-a.T).max()),
                            'min_eigenvalue_symmetrized':float(np.linalg.eigvalsh((a+a.T)/2).min()),
                            'mirroring_discrepancy':float(abs(a-mirrored).max())})
    dump('results/continuation/noise_validity.json',{'configuration':{'dataset':'wine','seed':0,'n':12,'qubits':4,
          'theta_source':'results/continuation/wine/seed_000.json','symmetric_flag':False},
          'rows':out,'warning':'One small matrix per channel/level; positive eigenvalues do not establish a general PSD theorem.'})
    print('maximum asymmetry',max(r['max_asymmetry_unmirrored'] for r in out))
if __name__=='__main__':run()
