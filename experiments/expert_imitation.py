"""Auditable 25-seed Expert-Imitation continuation. See frozen JSON protocol."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from sklearn.svm import SVC
from data.fold_preprocessing import raw_split, AnglePreprocessor
from models.adaptive_qkernel import AdaptiveQuantumKernel
from training.expert_imitation import oof_expert_decisions, make_targets, imitation_loss
from training.alignment import alignment_loss
from training.classification_loss import classification_surrogate_loss
from evaluation.stats import paired_comparison, holm_correction, capture_rate

ROOT=Path(__file__).resolve().parents[1]


def dump(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def source_hash():
    paths=[ROOT/'data/fold_preprocessing.py',ROOT/'data/datasets.py',ROOT/'models/adaptive_qkernel.py',
           ROOT/'quantum/kernels.py',ROOT/'quantum/feature_maps.py',ROOT/'routing/sample_level_router.py',
           ROOT/'training/expert_imitation.py',ROOT/'training/alignment.py',ROOT/'training/classification_loss.py',
           ROOT/'experiments/expert_imitation.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),
            'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','torch','pennylane','matplotlib','pytest']}}


def combine(g,h,k):
    return np.einsum('im,jm,ijm->ij',g,h,k)


def train_gate(initial_router,X,y,kt,objective,cfg,targets=None,weights=None):
    router=copy.deepcopy(initial_router)
    opt=torch.optim.Adam(router.parameters(),lr=cfg['router_lr'])
    kernels=torch.as_tensor(kt,dtype=torch.float32)
    losses=[];grad_norms=[]
    for _ in range(cfg['router_epochs']):
        opt.zero_grad()
        logits=router.net(torch.as_tensor(X,dtype=torch.float32))
        g=logits.softmax(-1)
        if objective in ('hard','soft','margin_weighted'):
            loss=imitation_loss(logits,targets,weights)
        else:
            K=(g[:,None,:]*g[None,:,:]*kernels).sum(-1)
            loss=alignment_loss(K,y) if objective=='alignment_fixed' else classification_surrogate_loss(K,y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad_norms.append(float(torch.sqrt(sum(p.grad.square().sum() for p in router.parameters()))))
        opt.step();losses.append(float(loss.detach()))
    return router,losses,grad_norms


def gate_prediction(g,h,kt,ke,y,C=1):
    k=combine(g,g,kt);q=combine(h,g,ke)
    clf=SVC(kernel='precomputed',C=C).fit(k,y)
    return clf.predict(q),clf.predict(k),k,q


def run_one(cfg,seed):
    torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    t0=time.monotonic()
    X,Z,y,yt,ids,ids_t=raw_split(cfg['dataset'],seed,cfg['max_train'],cfg['max_test'],cfg['test_size'])
    oof,folds=oof_expert_decisions(X,y,cfg,seed)
    prep=AnglePreprocessor(cfg['n_qubits'],seed).fit(X)
    x,z=prep.transform(X),prep.transform(Z)
    model=AdaptiveQuantumKernel(cfg['n_qubits'],seed=seed,kernel_backend=cfg['kernel_backend'])
    initial=copy.deepcopy(model.router)
    joint_losses=model.fit(x,y,epochs=cfg['teacher_epochs'],lr=cfg['teacher_lr'],verbose=False)
    kt=np.stack([model.single_kernel_matrix(n,x,x,symmetric=True) for n in model.kernel_names],axis=-1)
    ke=np.stack([model.single_kernel_matrix(n,z,x) for n in model.kernel_names],axis=-1)
    pred_tr=[];pred_te=[];dec_tr=[];dec_te=[]
    for m in range(3):
        svm=SVC(kernel='precomputed',C=cfg['svm_C']).fit(kt[:,:,m],y)
        pred_tr.append(svm.predict(kt[:,:,m]));pred_te.append(svm.predict(ke[:,:,m]))
        dec_tr.append(svm.decision_function(kt[:,:,m]));dec_te.append(svm.decision_function(ke[:,:,m]))
    pred_tr,pred_te,dec_tr,dec_te=[np.stack(v,axis=1) for v in [pred_tr,pred_te,dec_tr,dec_te]]
    oof_acc=((oof>0)==(y[:,None]>0)).mean(0)
    chosen=int(np.argmax(oof_acc))
    methods={}
    for variant in ['alignment_joint','alignment_fixed','classification_fixed']+cfg['variants']:
        target,weight=(None,None)
        if variant=='alignment_joint':
            router=model.router;losses=joint_losses;norms=[]
        else:
            if variant in cfg['variants']:target,weight=make_targets(oof,y,variant)
            router,losses,norms=train_gate(initial,x,y,kt,variant,cfg,target,weight)
        with torch.no_grad():
            g=router.gate_matrix(x).numpy();h=router.gate_matrix(z).numpy()
        pred,ptr,k,q=gate_prediction(g,h,kt,ke,y,cfg['svm_C'])
        # Mechanism interventions are pre-specified diagnostic outputs, not candidate selection.
        diag_tr=np.diag(k).clip(1e-12);diag_te=(h*h).sum(1).clip(1e-12)
        kn=k/np.sqrt(diag_tr[:,None]*diag_tr[None,:]);qn=q/np.sqrt(diag_te[:,None]*diag_tr[None,:])
        normalized=SVC(kernel='precomputed',C=cfg['svm_C']).fit(kn,y).predict(qn)
        hard=pred_te[np.arange(len(z)),h.argmax(1)]
        soft_vote=np.where((h*pred_te).sum(1)>0,1.,-1.)
        methods[variant]={'predictions':pred.tolist(),'train_predictions':ptr.tolist(),
                'gates_train':g.tolist(),'gates_test':h.tolist(), 'losses':losses, 'gradient_norms':norms,
                'gate_state':{n:v.tolist() for n,v in router.state_dict().items()},
                'normalized_predictions':normalized.tolist(),'hard_expert_predictions':hard.tolist(),
                'soft_vote_predictions':soft_vote.tolist(),
                'min_gram_eigenvalue':float(np.linalg.eigvalsh((k+k.T)/2).min()),
                'targets':None if target is None else target.tolist(),
                'target_weights':None if weight is None else weight.tolist()}
    # Scoring/test-label-informed diagnostics begin ONLY after deployable predictions are fixed.
    singles=(pred_te==yt[:,None]).mean(0)
    for v,r in methods.items():
        r['accuracy']=float(np.mean(np.array(r['predictions'])==yt))
        r['train_accuracy']=float(np.mean(np.array(r['train_predictions'])==y))
        for key in ['normalized','hard_expert','soft_vote']:
            r[key+'_accuracy']=float(np.mean(np.array(r[key+'_predictions'])==yt))
        h=np.array(r['gates_test']);correct=(pred_te==yt[:,None]);oracle=correct.any(1)
        r['normalized_entropy']=float(-(h*np.log(h.clip(1e-30))).sum(1).mean()/np.log(3))
        r['gate_precision_on_oracle']=float(correct[np.arange(len(z)),h.argmax(1)][oracle].mean()) if oracle.any() else None
        r['uniform_precision_on_oracle']=float(correct.mean(1)[oracle].mean()) if oracle.any() else None
    hindsight={}
    for variant in cfg['variants']:
        g,_=make_targets(dec_tr,y,variant);h,_=make_targets(dec_te,yt,variant)
        pred,_,_,_=gate_prediction(g,h,kt,ke,y,cfg['svm_C'])
        goof,_=make_targets(oof,y,variant)
        pred_oof,_,_,_=gate_prediction(goof,h,kt,ke,y,cfg['svm_C'])
        hindsight[variant]={'accuracy':float((pred==yt).mean()),'predictions':pred.tolist(),
                             'oof_train_gate_accuracy':float((pred_oof==yt).mean()),
                             'oof_train_gate_predictions':pred_oof.tolist()}
    return {'seed':seed,'dataset':cfg['dataset'],'train_ids':ids.tolist(),'test_ids':ids_t.tolist(),
            'raw_train_sha256':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
            'preprocessing':prep.state(),'folds':folds,'oof_decisions':oof.tolist(),'oof_expert_accuracy':oof_acc.tolist(),
            'y_train':y.tolist(),'y_test':yt.tolist(),'X_train':x.tolist(),'X_test':z.tolist(),
            'theta':model.theta.detach().tolist(),'kernel_names':model.kernel_names,
            'kernels_train':kt.tolist(),'kernels_test':ke.tolist(),
            'expert_predictions_train':pred_tr.tolist(),'expert_predictions_test':pred_te.tolist(),
            'expert_decisions_train':dec_tr.tolist(),'expert_decisions_test':dec_te.tolist(),
            'single_accuracies':singles.tolist(),'selected_single_index':chosen,
            'selected_single':float(singles[chosen]),'best_single_test_hindsight':float(singles.max()),
            'oracle':float((pred_te==yt[:,None]).any(1).mean()),
            'perfect_router':hindsight['margin_weighted']['accuracy'],'hindsight_gate_variants':hindsight,
            'methods':methods,'elapsed_seconds':time.monotonic()-t0}


def analyze(rows,cfg):
    rows=sorted(rows,key=lambda r:r['seed'])
    if [r['seed'] for r in rows]!=sorted(cfg['seeds']):raise ValueError('Incomplete or duplicate seed manifest')
    acc={v:[r['methods'][v]['accuracy'] for r in rows] for v in rows[0]['methods']}
    comparisons={}
    for comparator in ['alignment_joint','alignment_fixed','classification_fixed']:
        tests={v:paired_comparison(acc[v],acc[comparator],n_boot=cfg['bootstrap_draws'],seed=cfg['bootstrap_seed'],
                                   test_train_ratio=cfg['max_test']/cfg['max_train']) for v in cfg['variants']}
        for key in ['p_ttest','p_wilcoxon','p_corrected_resampled_t']:
            h=holm_correction({v:r[key] for v,r in tests.items()},cfg['alpha'])
            for v in tests:tests[v][key+'_holm']=h[v]['p_adjusted']
        comparisons[comparator]=tests
    # Conservative combined family across all 3 comparator families, supplied for readers selecting among controls.
    h9=holm_correction({f'{c}/{v}':r['p_ttest'] for c,t in comparisons.items() for v,r in t.items()})
    captures={base:{v:capture_rate(a,[r[base] for r in rows],[r['perfect_router'] for r in rows],
                                  n_boot=cfg['bootstrap_draws'],seed=cfg['bootstrap_seed']) for v,a in acc.items()}
              for base in ['selected_single','best_single_test_hindsight']}
    winners=[]
    for v,r in comparisons[cfg['primary_comparator']].items():
        if (r['mean_diff']>=cfg['minimum_mean_improvement'] and r['ci95_lo']>0 and
            all(r[k+'_holm']<=cfg['alpha'] for k in ['p_ttest','p_wilcoxon','p_corrected_resampled_t']) and
            comparisons['alignment_fixed'][v]['mean_diff']>0):winners.append(v)
    winner=max(winners,key=lambda v:np.mean(acc[v])) if winners else None
    return {'config':cfg,'n':len(rows),'means':{v:float(np.mean(a)) for v,a in acc.items()},
            'comparisons':comparisons,'all_nine_ttest_holm':h9,'capture_rates':captures,
            'decision':{'expand':bool(winners),'best_variant':winner,'passing_variants':winners,
                        'rule':cfg['decision_rule'],'status':'exploratory even if gate passes'},
            'baselines':{k:float(np.mean([r[k] for r in rows])) for k in ['selected_single','best_single_test_hindsight','perfect_router','oracle']}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/expert_imitation_wine.json')
    p.add_argument('--out',default='results/continuation/wine');p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out)
    meta={'config':cfg,'source_hash':source_hash(),'environment':environment()}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old!=meta:raise ValueError('Refusing to mix configurations/source/environment in a resumed run')
    else:dump(out/'manifest.json',meta)
    for seed in cfg['seeds']:
        path=out/f'seed_{seed:03d}.json'
        if path.exists():continue
        if args.analyze_only:raise ValueError(f'Missing seed {seed}')
        row=run_one(cfg,seed);dump(path,row)
        print(f"seed {seed}: "+', '.join(f"{v}={r['accuracy']:.3f}" for v,r in row['methods'].items())+f" ({row['elapsed_seconds']:.1f}s)",flush=True)
    rows=[json.loads((out/f'seed_{s:03d}.json').read_text()) for s in cfg['seeds']]
    result=analyze(rows,cfg);dump(out/'analysis.json',result)
    print(json.dumps({'means':result['means'],'decision':result['decision']},indent=2))

if __name__=='__main__':main()
