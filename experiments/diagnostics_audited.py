"""Saved decomposition and genuinely nested best-expert predictability probes."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from data.fold_preprocessing import raw_split,AnglePreprocessor
from models.adaptive_qkernel import AdaptiveQuantumKernel
from training.expert_imitation import oof_expert_decisions,fit_fold_experts,make_targets
from experiments.expert_imitation import dump,environment,source_hash,gate_prediction
from evaluation.stats import paired_comparison,holm_correction


def nested_probe(X,y,cfg,seed):
    c={**cfg,'inner_folds':cfg['probe_inner_folds']}
    records=[]
    outer=StratifiedKFold(cfg['probe_outer_folds'],shuffle=True,random_state=seed)
    for fold,(fit,held) in enumerate(outer.split(X,y)):
        # ALL label construction for probe fitting happens inside this outer fit subset.
        d,inner=oof_expert_decisions(X[fit],y[fit],c,seed)
        target,valid=make_targets(d,y[fit],'hard');valid=valid.astype(bool)
        dh,meta=fit_fold_experts(X,y,fit,held,c,seed)
        qh,vh=make_targets(dh,y[held],'hard');vh=vh.astype(bool)
        prep=AnglePreprocessor(cfg['n_qubits'],seed).fit(X[fit])
        xf,xh=prep.transform(X[fit]),prep.transform(X[held])
        labels=target.argmax(1)[valid];truth=qh.argmax(1)
        preds={};warnings_saved=[]
        if len(labels)==0:
            records.append({'fold':fold,'status':'no_correct_training_expert_targets'});continue
        for name in cfg['probes']:
            if len(np.unique(labels))==1:
                pred=np.repeat(labels[0],len(held))
            else:
                model=(LogisticRegression(max_iter=2000) if name=='linear' else
                       MLPClassifier(hidden_layer_sizes=tuple(cfg['mlp_hidden']),max_iter=cfg['mlp_max_iter'],random_state=seed) if name=='mlp' else
                       DummyClassifier(strategy='most_frequent'))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(xf[valid],labels)
                warnings_saved.extend([{'probe':name,'message':str(w.message)} for w in caught])
                pred=model.predict(xh)
            preds[name]=pred.tolist()
        records.append({'fold':fold,'fit_indices':fit.tolist(),'held_indices':held.tolist(),
                        'inner_folds_relative_to_fit':inner,'eval_teacher':meta,
                        'train_target_labels':target.argmax(1).tolist(),'train_valid':valid.tolist(),
                        'eval_target_labels':truth.tolist(),'eval_valid':vh.tolist(),
                        'predictions':preds,'warnings':warnings_saved})
    scores={};valid_records=[r for r in records if 'predictions' in r]
    total=sum(sum(r['eval_valid']) for r in valid_records)
    for probe in cfg['probes']:
        hits=sum(np.sum((np.array(r['predictions'][probe])==r['eval_target_labels'])&r['eval_valid']) for r in valid_records)
        scores[probe]=float(hits/total) if total else None
    return {'folds':records,'scores':scores,'valid_eval_count':total,'total':len(y),
            'note':'Conditional best-margin-expert identity prediction on points with at least one OOF-correct expert. Not task-label accuracy.'}


def decomposition(x,z,y,yt,cfg,seed):
    m=AdaptiveQuantumKernel(cfg['n_qubits'],seed=seed,kernel_backend=cfg['kernel_backend'])
    losses=m.fit(x,y,epochs=cfg['teacher_epochs'],lr=cfg['teacher_lr'],verbose=False)
    kt=np.stack([m.single_kernel_matrix(n,x,x,symmetric=True) for n in m.kernel_names],-1)
    ke=np.stack([m.single_kernel_matrix(n,z,x) for n in m.kernel_names],-1)
    dt=[];de=[];preds=[]
    for j in range(3):
        svm=SVC(kernel='precomputed',C=cfg['svm_C']).fit(kt[:,:,j],y)
        dt.append(svm.decision_function(kt[:,:,j]));de.append(svm.decision_function(ke[:,:,j]));preds.append(svm.predict(ke[:,:,j]))
    dt,de,preds=[np.stack(a,1) for a in (dt,de,preds)]
    g=m.router.gate_matrix(x).detach().numpy();h=m.router.gate_matrix(z).detach().numpy()
    pred,_,_,_=gate_prediction(g,h,kt,ke,y,cfg['svm_C'])
    pg,_=make_targets(dt,y,'margin_weighted');ph,_=make_targets(de,yt,'margin_weighted')
    perfect,_,_,_=gate_prediction(pg,ph,kt,ke,y,cfg['svm_C'])
    return {'best_single_test_hindsight':float((preds==yt[:,None]).mean(0).max()),
            'learned_router':float((pred==yt).mean()),'perfect_router':float((perfect==yt).mean()),
            'oracle':float((preds==yt[:,None]).any(1).mean()),'learned_predictions':pred.tolist(),
            'perfect_predictions':perfect.tolist(),'expert_predictions':preds.tolist(),'y_test':yt.tolist(),
            'gates_train':g.tolist(),'gates_test':h.tolist(),'losses':losses,'theta':m.theta.detach().tolist(),
            'note':'Test-best, Perfect Router, and Oracle are hindsight diagnostics. No monotone ordering is guaranteed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/mechanistic_diagnostics.json');p.add_argument('--out',default='results/continuation/diagnostics')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    meta={'config':cfg,'environment':environment(),'core_source_hash':source_hash(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Incompatible checkpoint')
    else:dump(out/'manifest.json',meta)
    for dataset,nq in cfg['datasets'].items():
        c={**cfg,'n_qubits':nq}
        for seed in cfg['seeds']:
            path=out/f'{dataset}_{seed:03}.json'
            if path.exists():continue
            X,Z,y,yt,ids,idst=raw_split(dataset,seed,cfg['max_train'],cfg['max_test'])
            prep=AnglePreprocessor(nq,seed).fit(X)
            record={'dataset':dataset,'seed':seed,'train_ids':ids.tolist(),'test_ids':idst.tolist(),
                    'probe':nested_probe(X,y,c,seed),
                    'decomposition':decomposition(prep.transform(X),prep.transform(Z),y,yt,c,seed)}
            dump(path,record);print(dataset,seed,'complete',flush=True)
    output={};tests={}
    for dataset in cfg['datasets']:
        rows=[json.loads((out/f'{dataset}_{s:03}.json').read_text()) for s in cfg['seeds']]
        output[dataset]={'n':len(rows),'probe_mean':{p:float(np.mean([r['probe']['scores'][p] for r in rows])) for p in cfg['probes']},
                         'decomposition_mean':{k:float(np.mean([r['decomposition'][k] for r in rows])) for k in ['best_single_test_hindsight','learned_router','perfect_router','oracle']}}
        for probe in ['linear','mlp']:
            tests[dataset+'/'+probe]=paired_comparison([r['probe']['scores'][probe] for r in rows],[r['probe']['scores']['majority'] for r in rows],seed=20260918,n_boot=20000,test_train_ratio=.5)
    for key in ['p_ttest','p_wilcoxon']:
        hs=holm_correction({n:r[key] for n,r in tests.items()})
        for n in tests:tests[n][key+'_holm']=hs[n]['p_adjusted']
    dump(out/'analysis.json',{'datasets':output,'probe_comparisons':tests,'warning':'10 splits reused within datasets; small, exploratory probes cannot establish information-theoretic limits.'})
if __name__=='__main__':main()
