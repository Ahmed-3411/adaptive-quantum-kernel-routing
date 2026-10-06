"""Publication exports from saved JSON only. No fitting or invented values.

PNG 300 dpi, vector PDF and SVG; source hashes and per-panel plotted values
are exported alongside figures. All intervals are descriptive seed-bootstrap
percentile intervals unless the panel explicitly shows restart distributions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures'
COLORS=['#0072B2','#009E73','#CC79A7','#D55E00','#56B4E9','#E69F00']
LABELS={'alignment_joint':'Joint alignment','alignment_fixed':'Fixed-bank alignment','classification_fixed':'Fixed-bank classification',
        'hard':'Hard imitation','soft':'Soft imitation','margin_weighted':'Margin imitation'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'svg.fonttype':'none','axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True})
SOURCES={};VALUES={}


def read(rel):
    p=ROOT/rel;SOURCES[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())


def interval(a):
    a=np.asarray(a,float);rng=np.random.default_rng(20260918)
    b=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
    return float(a.mean()),*np.quantile(b,[.025,.975]).tolist()


def save(fig,name,caption,values=None):
    OUT.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig);VALUES[name]={'caption':caption,'values':values}


def architecture():
    fig,ax=plt.subplots(figsize=(10,5.7));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color='#EAF3F8',edge='#0072B2'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec=edge,lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b,color='#334155',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=color,lw=1.2,linestyle=style))
    ax.text(.1,5.8,'Adaptive Quantum Kernel Routing: audited training and evaluation',fontsize=14,weight='bold')
    box(.15,4.5,1.65,.65,'Outer training data\n40 observations')
    box(2.2,4.4,2.3,.9,'5-fold cross-fitting\nFold-local scaling / PCA\nAlignment teacher + 3 SVMs')
    box(5.0,4.4,2.0,.9,'Held-fold predictions\n+ held training labels\nHard / soft / margin\ntarget distributions')
    box(7.55,4.4,2.2,.9,'Gate MLP: 4 → 16 → 16 → 3\n100 Adam steps, lr = 0.01\nWeighted cross entropy')
    arrow((1.8,4.85),(2.2,4.85));arrow((4.5,4.85),(5,4.85));arrow((7,4.85),(7.55,4.85))
    box(.15,2.9,1.65,.8,'Training transform\n+ alignment teacher\n15 steps, lr = 0.05')
    box(2.2,2.9,2.3,.8,'Shared frozen kernel bank\nIQP / XY / re-uploading\n4-qubit exact simulation')
    box(5.0,2.9,2,.8,'Sample gates g(x)\nPSD kernel combination\nΣ gₘ(x) gₘ(x′) Kₘ(x,x′)')
    box(7.55,2.9,2.2,.8,'Training Gram → SVM\nC = 1\nFit on training labels only')
    arrow((.98,4.5),(.98,3.7));arrow((1.8,3.3),(2.2,3.3));arrow((4.5,3.3),(5,3.3));arrow((7,3.3),(7.55,3.3));arrow((8.65,4.4),(6.0,3.7))
    box(.15,1.3,1.65,.7,'Outer test features\n20 observations','#EEF7EE','#009E73')
    box(2.2,1.3,2.3,.7,'Apply training transforms\nCompute gates + cross-Gram','#EEF7EE','#009E73')
    box(5.0,1.3,2,.7,'SVM predictions\nNo test-label input','#EEF7EE','#009E73')
    box(7.55,1.2,2.2,.9,'Final scoring + diagnostics\nTest-best / Perfect / Oracle\nTest labels allowed here only','#FFF0EA','#D55E00')
    arrow((1.8,1.65),(2.2,1.65));arrow((4.5,1.65),(5,1.65));arrow((7,1.65),(7.55,1.65));arrow((8.65,2.9),(6.0,2.0))
    ax.text(.15,.55,'Controls: original joint alignment; fixed-bank alignment and classification with the same gate-training budget.',fontsize=9)
    ax.text(.15,.18,'Hindsight diagnostics do not feed training, checkpoint selection, or targets.',fontsize=9,color='#A44010')
    save(fig,'01_system_architecture','Implemented Wine protocol; schematic, not experimental data. No test labels feed training. Architecture labels refer to Wine (4 qubits).')


def main():
    OUT.mkdir(exist_ok=True)
    rows=[read(f'results/continuation/wine/seed_{s:03}.json') for s in range(25)]
    analysis=read('results/continuation/wine/analysis.json');mech=read('results/continuation/mechanism_analysis.json')
    architecture()
    # Diagnostic decomposition, not a guaranteed ordered chain.
    datasets={'Wine (25 splits)':{'best_single_test_hindsight':[r['best_single_test_hindsight'] for r in rows],
                'learned_router':[r['methods']['alignment_joint']['accuracy'] for r in rows],
                'perfect_router':[r['perfect_router'] for r in rows],'oracle':[r['oracle'] for r in rows]}}
    for d in ['xor','synth_hard']:
        rr=[read(f'results/continuation/diagnostics/{d}_{s:03}.json')['decomposition'] for s in range(10)]
        datasets[f'{d} (10 splits)']={k:[r[k] for r in rr] for k in ['best_single_test_hindsight','learned_router','perfect_router','oracle']}
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.6),sharey=True,layout='constrained');vals={}
    for ax,(d,data) in zip(axs,datasets.items()):
        vals[d]={k:interval(a) for k,a in data.items()}
        for i,(k,(m,lo,hi)) in enumerate(vals[d].items()):
            ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',color=COLORS[i],capsize=4,ms=7)
            ax.text(i,m+.03,f'{m:.3f}',ha='center',fontsize=8)
        ax.set_title(d);ax.set_xticks(range(4),['Best single*','Learned','Perfect*','Oracle*'],rotation=25,ha='right')
        ax.set_ylim(.5,1.035);ax.yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_ylabel('Test accuracy');fig.suptitle('Best Single → Learned → Perfect → Oracle\n*Hindsight diagnostics; arrows do not imply guaranteed ordering',fontsize=12)
    save(fig,'02_router_decomposition','Pointwise 95% seed-bootstrap intervals; splits reuse data. Learned=joint alignment. Perfect=margin-weighted in-sample-train and test-label-informed gates. Best single is test-selected hindsight.',vals)
    # Capture rates, both baselines explicit.
    methods=['alignment_joint','alignment_fixed','classification_fixed','hard','soft','margin_weighted']
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained');vals={}
    for ax,base,title in zip(axs,['best_single_test_hindsight','selected_single'],['Relative to test-best single (hindsight)','Relative to OOF-selected single']):
        vals[base]={}
        for i,v in enumerate(methods):
            r=analysis['capture_rates'][base][v];m,lo,hi=r['estimate'],r['ci95_lo'],r['ci95_hi'];vals[base][v]=r
            if m is not None:
                ax.plot(m,i,'o',color=COLORS[i])
                if lo is not None:ax.plot([lo,hi],[i,i],color=COLORS[i],lw=2)
        ax.axvline(0,color='.4',lw=.8);ax.axvline(1,color='.6',ls=':',lw=.8)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Ratio of mean gain to mean Perfect-Router gap');ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_yticks(range(6),[LABELS[v] for v in methods])
    axs[0].invert_yaxis();fig.suptitle('Wine: capture of a heuristic hindsight reference gap',fontsize=13)
    save(fig,'03_capture_rates','Ratios of means with paired seed-bootstrap intervals. Not bounded efficiencies; no guaranteed ceiling. Same 25 seeds for every numerator/denominator.',vals)
    # Accuracy and primary paired differences.
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');vals={}
    for i,v in enumerate(methods):
        m,lo,hi=interval([r['methods'][v]['accuracy'] for r in rows]);vals[v]={'accuracy':[m,lo,hi]}
        axs[0].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=COLORS[i])
    axs[0].set_yticks(range(6),[LABELS[v] for v in methods]);axs[0].invert_yaxis();axs[0].set_xlabel('Test accuracy');axs[0].xaxis.set_major_formatter(PercentFormatter(1))
    for i,v in enumerate(['hard','soft','margin_weighted']):
        r=analysis['comparisons']['alignment_joint'][v];m=r['mean_diff'];lo=r['ci95_lo'];hi=r['ci95_hi'];vals[v]['difference']=r
        axs[1].errorbar(m*100,i,xerr=[[(m-lo)*100],[(hi-m)*100]],fmt='o',capsize=4,color=COLORS[i+3])
        axs[1].text(.98,.9-i*.31,f"Holm p: t={r['p_ttest_holm']:.3f}, W={r['p_wilcoxon_holm']:.3f}",transform=axs[1].transAxes,ha='right',fontsize=8)
    axs[1].axvline(0,color='.4',lw=1);axs[1].set_yticks(range(3),['Hard','Soft','Margin']);axs[1].invert_yaxis();axs[1].set_ylim(2.5,-.55);axs[1].set_xlabel('Accuracy difference vs joint alignment (pp)')
    fig.suptitle('Wine: Expert-Imitation does not improve accuracy (25 paired splits)',fontsize=13)
    save(fig,'04_objective_comparison','Mean accuracy and nominal 95% seed-bootstrap intervals; primary differences are paired. Holm adjusts three variants separately for each test. Repeated-split population independence is not assumed.',vals)
    # Historical noise data only, pointwise descriptive intervals.
    noise=read('results/noise_wine_psd_raw.json');levels=sorted(map(float,noise));fig,axs=plt.subplots(1,2,figsize=(9,3.6),layout='constrained');vals={}
    for key,color,label in [('single',COLORS[0],'Ideal-test-selected single (legacy)'),('adaptive',COLORS[1],'Adaptive (legacy)')]:
        a=np.array([interval(noise[str(l)][key]) for l in levels]);vals[key]=a.tolist()
        axs[0].plot(levels,a[:,0],'-o',color=color,label=label);axs[0].fill_between(levels,a[:,1],a[:,2],color=color,alpha=.15)
    a=np.array([interval(np.array(noise[str(l)]['adaptive'])-noise[str(l)]['single']) for l in levels]);vals['paired_difference']=a.tolist()
    axs[1].errorbar(levels,100*a[:,0],yerr=100*np.array([a[:,0]-a[:,1],a[:,2]-a[:,0]]),fmt='-o',color=COLORS[3],capsize=3)
    axs[1].axhline(0,color='.5',lw=.8)
    for ax in axs:ax.set_xlabel('Depolarizing channel probability')
    axs[0].set_ylabel('Test accuracy');axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[0].legend(fontsize=7,loc='lower left')
    axs[1].set_ylabel('Adaptive − single (percentage points)')
    fig.suptitle('Historical Wine noise sweep: 25 splits, descriptive intervals',fontsize=13)
    save(fig,'05_noise_robustness','Actual legacy arrays; test-selected baseline, incomplete environment provenance, noisy overlap lacks general PSD guarantee. Difference-in-gap from noise 0 to .1 is not significant (nominal t p=.185). Bands are pointwise and not simultaneous.',vals)
    # Gate distributions and entropy, per-seed summaries for inference.
    fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained');vals={}
    for ax,v,color in zip(axs.flat,methods,COLORS):
        g=np.concatenate([r['methods'][v]['gates_test'] for r in rows]);vals[v]={'mean_gate':g.mean(0).tolist()}
        for m in range(3):ax.hist(g[:,m],bins=np.linspace(0,1,16),histtype='step',density=True,label=f'K{m+1}',color=COLORS[m],lw=1.5)
        ax.set_title(LABELS[v]);ax.set_xlabel('Gate weight');ax.set_ylabel('Density')
    axs[0,0].legend(fontsize=8);fig.suptitle('Wine test-gate distributions (pooled for visualization only)',fontsize=13)
    save(fig,'06_gate_distributions','500 test appearances per method; points can recur across splits, so these histograms are descriptive, not independent observations.',vals)
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained');vals={}
    for i,v in enumerate(methods):
        a=np.array([r['methods'][v]['normalized_entropy'] for r in rows]);m,lo,hi=interval(a);vals[v]={'mean_ci':[m,lo,hi],'seed_values':a.tolist()}
        ax.scatter(i+np.linspace(-.16,.16,len(a)),a,s=12,alpha=.5,color=COLORS[i]);ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='D',color='black',capsize=4,ms=5)
    ax.set_xticks(range(6),[LABELS[v] for v in methods],rotation=20,ha='right');ax.set_ylabel('Mean gate entropy / log(3)');ax.set_title('Wine gate entropy: one mean per split');ax.set_ylim(-.03,1.03)
    save(fig,'07_gate_entropy','Dots are per-seed mean normalized entropy; black diamonds and bars are means and descriptive bootstrap intervals.',vals)
    # Mechanism plot.
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained');vals=mech['intervention_means']
    for j,(key,label) in enumerate([('combined','Original Gram'),('normalized','Normalized Gram'),('hard_expert','Top-1 expert'),('soft_vote','Soft vote')]):
        ax.plot(range(6),[vals[v][key] for v in methods],'-o',label=label,color=COLORS[j])
    ax.set_xticks(range(6),[LABELS[v] for v in methods],rotation=20,ha='right');ax.set_ylabel('Mean test accuracy');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=2,fontsize=8)
    ax.set_title('Wine: identical gates, different prediction mechanisms')
    save(fig,'08_mechanism_interventions','Descriptive paired means. Full 18-comparison corrected tests in mechanism_analysis.json; normalization gains do not survive that family.',vals)
    # Stability charts show all restarts within fixed data splits.
    stability=read('results/continuation/stability/analysis.json');cfg=stability['config']
    for mode in ['joint','gate_only']:
        fig,axs=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True,layout='constrained');vals={}
        for row,d in enumerate(['xor','synth_hard']):
            for col,ds in enumerate(cfg['data_seeds']):
                ax=axs[row,col]
                for j,ep in enumerate(cfg['epoch_budgets']):
                    means=[]
                    for i,lr in enumerate(cfg['learning_rates']):
                        rr=[read(f'results/continuation/stability/{d}_split{ds}_{mode}_lr{lr}_ep{ep}_init{s}.json') for s in cfg['training_seeds']]
                        a=np.array([r['accuracy'] for r in rr]);vals[f'{d}/{ds}/{lr}/{ep}']=a.tolist();means.append(a.mean())
                        ax.scatter(i+(.09 if j else -.09)+np.linspace(-.04,.04,8),a,s=15,alpha=.65,color=COLORS[j])
                        ax.plot([i+(.09 if j else -.09)]*2,[a.min(),a.max()],color=COLORS[j],alpha=.45,lw=1)
                    ax.plot(np.arange(3)+(.09 if j else -.09),means,'-D',color=COLORS[j],label=f'{ep} epochs',ms=4)
                ax.set_title(f'{d} · data split {ds}');ax.set_xticks(range(3),['0.01','0.05','0.10']);ax.set_ylim(.25,1.02)
                if row==1:ax.set_xlabel('Learning rate')
                if col==0:ax.set_ylabel('Test accuracy')
        axs[0,0].legend(fontsize=8);fig.suptitle(f'Optimization stability: {mode.replace("_"," ")} · 8 restarts per setting',fontsize=13)
        save(fig,f'09_stability_{mode}','Dots=all restarts; diamonds=mean; vertical lines=min/max, NOT confidence intervals. Each panel holds the data split fixed. No best test restart selection.',vals)
    # Properly nested probes.
    diag=read('results/continuation/diagnostics/analysis.json');fig,ax=plt.subplots(figsize=(7.5,3.8),layout='constrained');vals={}
    for j,p in enumerate(['linear','mlp','majority']):
        for i,d in enumerate(['wine','xor','synth_hard']):
            a=[read(f'results/continuation/diagnostics/{d}_{s:03}.json')['probe']['scores'][p] for s in range(10)]
            m,lo,hi=interval(a);vals[f'{d}/{p}']=[m,lo,hi]
            ax.errorbar(i+(j-1)*.2,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=COLORS[j],label=p if i==0 else None)
    ax.set_xticks(range(3),['Wine','XOR','synth_hard']);ax.set_ylabel('Conditional expert-identity accuracy');ax.set_ylim(0,1);ax.legend(ncol=3);ax.set_title('Nested predictability probes: 10 outer training splits')
    save(fig,'10_nested_predictability','Each probe fold regenerates training targets by an inner CV entirely within its fit subset. Held probe targets come from experts fitted only on the fit subset. Conditional metric excludes rows with no correct expert; majority baseline reported. Intervals are descriptive.',vals)
    dump={'source_sha256':SOURCES,'figures':VALUES,'bootstrap':{'draws':20000,'seed':20260918},'formats':['png (300 dpi)','pdf (vector)','svg (vector)']}
    (OUT/'figure_manifest.json').write_text(json.dumps(dump,indent=2,allow_nan=False))
    (OUT/'CAPTIONS.md').write_text('# Figure captions and provenance\n\n'+'\n\n'.join(f'## {n}\n\n{v["caption"]}' for n,v in VALUES.items()))
    print(f'Generated {len(VALUES)} figures in PNG/PDF/SVG from {len(SOURCES)} saved source files.')

if __name__=='__main__':main()
