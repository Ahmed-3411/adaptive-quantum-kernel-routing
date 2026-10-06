"""Analyze pre-specified same-gate interventions from saved Wine predictions."""
import json
from pathlib import Path
import numpy as np
from evaluation.stats import paired_comparison,holm_correction
from experiments.expert_imitation import dump


def run():
    root=Path('results/continuation/wine');rows=[json.loads(p.read_text()) for p in sorted(root.glob('seed_*.json'))]
    tests={};summaries={};diagnostics={}
    for v in rows[0]['methods']:
        original=[r['methods'][v]['accuracy'] for r in rows]
        summaries[v]={'combined':float(np.mean(original))}
        for mechanism in ['normalized','hard_expert','soft_vote']:
            a=[r['methods'][v][mechanism+'_accuracy'] for r in rows]
            tests[v+'/'+mechanism]=paired_comparison(a,original,n_boot=20000,seed=20260918,test_train_ratio=.5)
            summaries[v][mechanism]=float(np.mean(a))
        diagnostics[v]={k:float(np.mean([r['methods'][v][k] for r in rows])) for k in
                        ['normalized_entropy','gate_precision_on_oracle','uniform_precision_on_oracle']}
        if v in ('hard','soft','margin_weighted'):
            # Evaluate imitation on training targets and on full-bank test correctness as distinct diagnostics.
            train_hits=[];train_support=[];test_hits=[];coverage=[];correct_weight=[];rescued=[];target_mix=[]
            for r in rows:
                m=r['methods'][v];g=np.array(m['gates_train']);h=np.array(m['gates_test'])
                q=np.array(m['targets']);valid=np.array(m['target_weights'])>0
                train_hits.append(float((q.argmax(1)[valid]==g.argmax(1)[valid]).mean()))
                train_support.append(float((q[np.arange(len(q)),g.argmax(1)][valid]>0).mean()))
                test_correct=np.array(r['expert_predictions_test'])==np.array(r['y_test'])[:,None]
                oracle=test_correct.any(1);coverage.append(float(valid.mean()))
                test_hits.append(float(test_correct[np.arange(len(h)),h.argmax(1)].mean()))
                correct_weight.append(float((h*test_correct).sum(1).mean()))
                target_mix.append(q[valid].mean(0).tolist())
                pred=np.array(m['predictions']);rescuable=oracle&(pred!=r['y_test'])
                rescued.append(float(rescuable.mean()))
            diagnostics[v].update(target_top1_train_accuracy=float(np.mean(train_hits)),
                         target_top1_note='Exact target argmax agreement; arbitrary for uniform-soft ties',
                         chosen_expert_in_train_target_support=float(np.mean(train_support)),
                         oof_valid_fraction=float(np.mean(coverage)),top1_expert_test_accuracy=float(np.mean(test_hits)),
                         test_gate_mass_on_correct_experts=float(np.mean(correct_weight)),
                         oracle_rescuable_failure_fraction=float(np.mean(rescued)),
                         target_mean_mix=np.mean(target_mix,axis=0).tolist())
    for key in ['p_ttest','p_wilcoxon','p_corrected_resampled_t']:
        h=holm_correction({n:r[key] for n,r in tests.items()})
        for n in tests:tests[n][key+'_holm_18']=h[n]['p_adjusted']
    perfect={v:{k:float(np.mean([r['hindsight_gate_variants'][v][k] for r in rows])) for k in
                         ['accuracy','oof_train_gate_accuracy']} for v in ['hard','soft','margin_weighted']}
    out={'n':len(rows),'intervention_means':summaries,'paired_interventions':tests,
         'diagnostics':diagnostics,'hindsight_heuristics':perfect,
         'claims':['A comparison changes only the decoding/combination rule for the same stored gates and bank.',
             'Diagonal normalization changes geometry/scale and effective regularization; not a pure routing intervention.',
             'Hindsight gates use test labels; they cannot establish a deployable accuracy or optimal mechanism ceiling.',
             'All 18 decoding comparisons form one exploratory Holm family; no winning decoder chosen for deployment.',
             'High train-target agreement with poor test expert selection is consistent with limited generalization and/or noisy teacher targets, not proof of a unique cause.']}
    dump('results/continuation/mechanism_analysis.json',out)
    print(json.dumps({'means':summaries,'diagnostics':diagnostics,'hindsight':perfect},indent=2))
if __name__=='__main__':run()
