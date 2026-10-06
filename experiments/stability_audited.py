"""Fixed-split, saved optimization diagnostics. No test-based model selection."""
import argparse
import copy
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from sklearn.svm import SVC
from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from training.alignment import alignment_loss
from experiments.expert_imitation import dump,environment,source_hash,train_gate,combine


def run(cfg,out):
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    manifest={'config':cfg,'environment':environment(),'core_source_hash':source_hash(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Incompatible checkpoint')
    else:dump(out/'manifest.json',manifest)
    for dataset,nq in cfg['datasets'].items():
        for ds in cfg['data_seeds']:
            x,z,y,yt=load_dataset(dataset,n_qubits=nq,seed=ds,max_train=cfg['max_train'],max_test=cfg['max_test'])
            bank=AdaptiveQuantumKernel(nq,seed=ds,kernel_backend=cfg['kernel_backend'])
            bank.fit(x,y,epochs=15,lr=.05,verbose=False)
            ft=np.stack([bank.single_kernel_matrix(n,x,x,symmetric=True) for n in bank.kernel_names],-1)
            fe=np.stack([bank.single_kernel_matrix(n,z,x) for n in bank.kernel_names],-1)
            for mode in cfg['modes']:
                for lr in cfg['learning_rates']:
                    for epochs in cfg['epoch_budgets']:
                        for ts in cfg['training_seeds']:
                            path=out/f'{dataset}_split{ds}_{mode}_lr{lr}_ep{epochs}_init{ts}.json'
                            if path.exists():continue
                            model=AdaptiveQuantumKernel(nq,seed=ts,kernel_backend=cfg['kernel_backend'])
                            if mode=='joint':
                                losses=model.fit(x,y,epochs=epochs,lr=lr,verbose=False)
                                kt=model.kernel_matrix(x,x,symmetric=True);ke=model.kernel_matrix(z,x)
                            else:
                                c={'router_lr':lr,'router_epochs':epochs}
                                router,losses,_=train_gate(model.router,x,y,ft,'alignment_fixed',c)
                                model.router=router;model.theta.data.copy_(bank.theta.data)
                                with torch.no_grad():g=router.gate_matrix(x).numpy();h=router.gate_matrix(z).numpy()
                                kt=combine(g,g,ft);ke=combine(h,g,fe)
                            svm=SVC(kernel='precomputed',C=cfg['svm_C']).fit(kt,y)
                            pred=svm.predict(ke);ptr=svm.predict(kt)
                            g=model.router.gate_matrix(z).detach().numpy()
                            final_loss=float(alignment_loss(torch.tensor(kt),y))
                            dump(path,{'dataset':dataset,'data_seed':ds,'training_seed':ts,'mode':mode,'lr':lr,'epochs':epochs,
                                'accuracy':float((pred==yt).mean()),'train_accuracy':float((ptr==y).mean()),
                                'predictions':pred.tolist(),'y_test':yt.tolist(),'losses':losses,'final_loss':final_loss,
                                'gates_test':g.tolist(),'theta':model.theta.detach().tolist(),
                                'normalized_entropy':float(-(g*np.log(g.clip(1e-30))).sum(1).mean()/np.log(3)),
                                'data_sha256':hashlib.sha256(x.tobytes()+z.tobytes()+y.tobytes()+yt.tobytes()).hexdigest()})
                        print(f'{dataset} split={ds} mode={mode} lr={lr} epochs={epochs}: complete',flush=True)
    rows=[json.loads(p.read_text()) for p in out.glob('*_init*.json')]
    summary=[]
    for dataset in cfg['datasets']:
        for ds in cfg['data_seeds']:
            for mode in cfg['modes']:
                for lr in cfg['learning_rates']:
                    for epochs in cfg['epoch_budgets']:
                        group=[r for r in rows if (r['dataset'],r['data_seed'],r['mode'],r['lr'],r['epochs'])==(dataset,ds,mode,lr,epochs)]
                        if len(group)!=len(cfg['training_seeds']):raise ValueError('Incomplete restart group')
                        acc=np.array([r['accuracy'] for r in group])
                        summary.append({'dataset':dataset,'data_seed':ds,'mode':mode,'lr':lr,'epochs':epochs,
                              'n':len(acc),'mean':float(acc.mean()),'sd':float(acc.std(ddof=1)),
                              'min':float(acc.min()),'max':float(acc.max()),'range':float(np.ptp(acc)),
                              'mean_final_loss':float(np.mean([r['final_loss'] for r in group]))})
    dump(out/'analysis.json',{'config':cfg,'n_runs':len(rows),'groups':summary})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/optimization_stability.json');p.add_argument('--out',default='results/continuation/stability')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),Path(a.out))
