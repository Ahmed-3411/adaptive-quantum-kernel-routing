"""Integrity/provenance verification and deterministic replay of stored predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.svm import SVC
from routing.sample_level_router import SampleLevelGate
from training.expert_imitation import make_targets
from experiments.expert_imitation import ROOT,source_hash,analyze,dump,combine


def read(p):
    def reject(v):raise ValueError('Nonstandard JSON constant: '+v)
    return json.loads(Path(p).read_text(),parse_constant=reject)


def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-10)
    else:assert a==b,(a,b)


def main():
    torch.set_num_threads(1)
    original=read(ROOT/'docs/input_manifest.json');input_results={p:h for p,h in original.items() if p.startswith('results/')}
    for rel,h in input_results.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    wp=ROOT/'results/continuation/wine';manifest=read(wp/'manifest.json');cfg=manifest['config']
    assert cfg==read(ROOT/'configs/expert_imitation_wine.json');assert manifest['source_hash']==source_hash()
    rows=[read(wp/f'seed_{s:03d}.json') for s in cfg['seeds']]
    for r in rows:
        y=np.array(r['y_train']);yt=np.array(r['y_test']);kt=np.array(r['kernels_train']);ke=np.array(r['kernels_test'])
        assert not set(r['train_ids'])&set(r['test_ids'])
        held=[]
        for f in r['folds']:
            fit=f['fit_indices'];h=f['held_indices'];assert not set(fit)&set(h)
            assert set(fit)|set(h)==set(range(len(y)));held+=h
        assert sorted(held)==list(range(len(y)))
        assert r['selected_single_index']==int(np.argmax(r['oof_expert_accuracy']))
        pp=np.array(r['expert_predictions_test']);single=(pp==yt[:,None]).mean(0)
        np.testing.assert_allclose(single,r['single_accuracies'])
        assert r['best_single_test_hindsight']==single.max()
        assert r['selected_single']==single[r['selected_single_index']]
        assert r['oracle']==(pp==yt[:,None]).any(1).mean()
        for v,m in r['methods'].items():
            assert m['accuracy']==np.mean(np.array(m['predictions'])==yt)
            if v in cfg['variants']:
                q,w=make_targets(r['oof_decisions'],y,v)
                np.testing.assert_allclose(q,m['targets'],atol=0);np.testing.assert_array_equal(w,m['target_weights'])
            gate=SampleLevelGate(cfg['n_qubits'],3)
            state={k:torch.tensor(a,dtype=torch.float32) for k,a in m['gate_state'].items()};gate.load_state_dict(state)
            with torch.no_grad():g=gate.gate_matrix(r['X_train']).numpy();h=gate.gate_matrix(r['X_test']).numpy()
            np.testing.assert_allclose(g,m['gates_train'],atol=1e-7);np.testing.assert_allclose(h,m['gates_test'],atol=1e-7)
            # Replay with original stored float32 gates, matching arithmetic in the experiment.
            g=np.array(m['gates_train'],np.float32);h=np.array(m['gates_test'],np.float32)
            k=combine(g,g,kt);q=combine(h,g,ke)
            pred=SVC(kernel='precomputed',C=cfg['svm_C']).fit(k,y).predict(q)
            np.testing.assert_array_equal(pred,m['predictions'])
            assert m['min_gram_eigenvalue']>=-1e-5
            np.testing.assert_allclose(g.sum(1),1,atol=1e-6)
        for v,d in r['hindsight_gate_variants'].items():
            assert d['accuracy']==np.mean(np.array(d['predictions'])==yt)
            assert d['oof_train_gate_accuracy']==np.mean(np.array(d['oof_train_gate_predictions'])==yt)
    recomputed=analyze(rows,cfg);equal(recomputed,read(wp/'analysis.json'))
    assert not recomputed['decision']['expand']
    legacy=read(ROOT/'results/oracle_wine_raw.json')
    assert [r['methods']['alignment_joint']['accuracy'] for r in rows]==legacy['adaptive']
    assert [r['best_single_test_hindsight'] for r in rows]==legacy['best_single']
    sp=ROOT/'results/continuation/stability';sm=read(sp/'manifest.json');assert sm['core_source_hash']==source_hash()
    stability=[read(p) for p in sp.glob('*_init*.json')];assert len(stability)==576
    identities=set();data_hashes={};fixed_theta={}
    for r in stability:
        identity=tuple(r[k] for k in ['dataset','data_seed','training_seed','mode','lr','epochs']);assert identity not in identities;identities.add(identity)
        assert r['accuracy']==np.mean(np.array(r['predictions'])==r['y_test'])
        assert len(r['losses'])==r['epochs'] and np.isfinite(r['losses']).all()
        key=(r['dataset'],r['data_seed']);data_hashes.setdefault(key,set()).add(r['data_sha256'])
        if r['mode']=='gate_only':
            if key in fixed_theta:assert fixed_theta[key]==r['theta']
            fixed_theta[key]=r['theta']
    assert all(len(v)==1 for v in data_hashes.values())
    dp=ROOT/'results/continuation/diagnostics';dm=read(dp/'manifest.json');assert dm['core_source_hash']==source_hash()
    diagnostics=[read(dp/f'{d}_{s:03}.json') for d in dm['config']['datasets'] for s in dm['config']['seeds']]
    assert len(diagnostics)==30
    for r in diagnostics:
        assert not set(r['train_ids'])&set(r['test_ids'])
        outerheld=[]
        for f in r['probe']['folds']:
            fit=np.array(f['fit_indices']);held=f['held_indices'];outerheld+=held
            assert not set(fit)&set(held)
            innerheld=[]
            for inn in f['inner_folds_relative_to_fit']:
                assert not set(inn['fit_indices'])&set(inn['held_indices'])
                assert not set(fit[inn['fit_indices']])&set(held)
                innerheld+=inn['held_indices']
            assert sorted(innerheld)==list(range(len(fit)))
        assert sorted(outerheld)==list(range(len(r['train_ids'])))
        d=r['decomposition'];yt=np.array(d['y_test'])
        assert d['learned_router']==np.mean(np.array(d['learned_predictions'])==yt)
        assert d['perfect_router']==np.mean(np.array(d['perfect_predictions'])==yt)
    fm=read(ROOT/'figures/figure_manifest.json')
    for p,h in fm['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    for name in fm['figures']:
        for ext in ['pdf','svg','png']:assert (ROOT/'figures'/f'{name}.{ext}').stat().st_size>0
    out={'status':'passed','original_result_files_unchanged':len(input_results),'wine_seeds':len(rows),
         'wine_methods_replayed_from_saved_gates_and_kernels':sum(len(r['methods']) for r in rows),
         'historical_wine_baseline_per_seed_exact_match':True,'stability_runs':len(stability),'nested_diagnostic_runs':len(diagnostics),
         'figures':len(fm['figures']),'source_manifests_match':True,'analysis_recomputed':True,
         'limits':'Verification establishes internal consistency and executable provenance, not statistical independence, external validity, or absence of every possible bug.'}
    dump(ROOT/'results/continuation/verification.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
