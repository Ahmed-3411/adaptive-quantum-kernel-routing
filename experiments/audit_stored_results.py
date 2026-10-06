"""Read ALL input result leaves; recalculate claims without rewriting historical data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from evaluation.stats import paired_comparison,holm_correction

ROOT=Path(__file__).resolve().parents[1]


def inventory_value(value, path=''):
    if isinstance(value,dict):
        return {'kind':'object','children':{k:inventory_value(v,path+'/'+k) for k,v in value.items()}}
    if isinstance(value,list):
        if value and all(isinstance(v,(int,float)) for v in value):
            a=np.array(value,float)
            return {'kind':'numeric_array','n':len(a),'finite':bool(np.isfinite(a).all()),
                    'mean':float(a.mean()),'sd_population':float(a.std()),'min':float(a.min()),'max':float(a.max())}
        return {'kind':'array','n':len(value),'items':[inventory_value(v,path+f'/{i}') for i,v in enumerate(value)]}
    return {'kind':'scalar','value':value}


def run():
    original=json.loads((ROOT/'docs/input_manifest.json').read_text())
    result_paths=[p for p in original if p.startswith('results/') and p.endswith('.json')]
    inventory={};canonical={};duplicates=[]
    for rel in result_paths:
        p=ROOT/rel;d=json.loads(p.read_text());sha=hashlib.sha256(p.read_bytes()).hexdigest()
        semantic=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
        if semantic in canonical:duplicates.append([canonical[semantic],rel])
        canonical[semantic]=rel
        inventory[rel]={'sha256':sha,'original_unchanged':sha==original[rel],
                        'explicit_config':isinstance(d,dict) and 'config' in d,
                        'content':inventory_value(d)}
    def read(name):return json.loads((ROOT/'results'/name).read_text())
    def pair(d,a='adaptive',b='single'):return paired_comparison(d[a],d[b],seed=20260918,n_boot=20000,test_train_ratio=.5)
    wine=read('oracle_wine_raw.json');hold=read('wine_holdout_seeds25_49.json')
    out={'n_input_result_files':len(result_paths),'inventory':inventory,'exact_semantic_duplicates':duplicates,
         'wine_original':pair(wine,b='best_single'),'wine_replication':pair(hold),
         'wine_combined':pair({'adaptive':wine['adaptive']+hold['adaptive'],'single':wine['best_single']+hold['single']}),
         'objective_B':pair(read('objective_compare_wine.json'),a='classification',b='alignment')}
    panel=read('all_datasets_psd.json');checks={n:pair(d) for n,d in panel.items()}
    out['historical_15_panel']=checks
    out['historical_15_t_holm']=holm_correction({n:r['p_ttest'] for n,r in checks.items()})
    out['historical_15_wilcoxon_holm']=holm_correction({n:r['p_wilcoxon'] for n,r in checks.items()})
    div=read('diversity_all_datasets.json')
    rho,p=spearmanr([div[n] for n in panel],[round(checks[n]['mean_diff'],12) for n in panel])
    out['diversity_correlation']={'tie_policy':'mean gains rounded to 12 decimals before ranking','rho':float(rho),'p':float(p),'n':len(panel),'warning':'Mixed seed counts; one diversity seed per dataset; exploratory'}
    rawrho,rawp=spearmanr([div[n] for n in panel],[float(np.mean(np.array(panel[n]['adaptive'])-panel[n]['single'])) for n in panel])
    out['diversity_correlation_unrounded_historical']={'rho':float(rawrho),'p':float(rawp),'warning':'Floating-point noise breaks mathematically equal accuracy-gain ties'}
    psd={n:pair(read(f'e2_psd_{n}.json')) for n in ['xor','moons','circles','digits','breast_cancer']}
    psd['wine']=out['wine_original'];out['later_six_psd']=psd
    out['later_six_psd_holm_t']=holm_correction({n:r['p_ttest'] for n,r in psd.items()})
    noise=read('noise_wine_psd_raw.json');out['noise']={n:pair(d) for n,d in noise.items()}
    for key in ['p_ttest','p_wilcoxon']:
        out['noise_'+key+'_holm']=holm_correction({n:r[key] for n,r in out['noise'].items()})
    first,last=noise['0.0'],noise['0.1']
    out['noise_gap_change']=paired_comparison(np.array(last['adaptive'])-last['single'],np.array(first['adaptive'])-first['single'],n_boot=20000,seed=20260918,test_train_ratio=.5)
    out['warnings']=['Most paired arrays have implicit positional pairing, not verified seed IDs.',
          'Stored summary SD uses ddof=0; continuation uses sample SD ddof=1.',
          'Stored Wilcoxon values can differ due to floating ties; continuation rounds differences.',
          'No cross-dataset PSD benefit survives a family requiring both Holm-adjusted t and Wilcoxon.',
          'Missing raw data: historical perfect-router table, nonlinear predictability table, Wine restart table.',
          'Original historical claims remain archived; all numerical arrays preserved byte-for-byte.']
    path=ROOT/'results/continuation/historical_audit.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:out[k] for k in ['n_input_result_files','exact_semantic_duplicates','diversity_correlation','wine_original','wine_replication','noise_gap_change']},indent=2))
if __name__=='__main__':run()
