import numpy as np
import torch
import pytest
from quantum.feature_maps import FEATURE_MAPS, TRAINABLE_THETA_SHAPE
from quantum.kernels import QuantumKernel
from training.expert_imitation import make_targets, imitation_loss, fit_fold_experts
from data.fold_preprocessing import raw_split, AnglePreprocessor
from data.datasets import load_dataset
from evaluation.stats import paired_comparison, holm_correction, capture_rate

torch.set_num_threads(1)

@pytest.mark.parametrize('name',list(FEATURE_MAPS))
def test_statevector_matches_overlap_values_and_gradients(name):
    rng=np.random.default_rng(12);X=rng.normal(size=(5,4));Y=rng.normal(size=(3,4))
    spec=FEATURE_MAPS[name]
    a=QuantumKernel(name,spec['fn'],4,spec['trainable'],backend='overlap')
    b=QuantumKernel(name,spec['fn'],4,spec['trainable'],backend='statevector')
    th=torch.tensor(rng.normal(size=(4,*TRAINABLE_THETA_SHAPE)),requires_grad=True)
    ka=a.matrix(X,Y,th if spec['trainable'] else None)
    kb=b.matrix(X,Y,th if spec['trainable'] else None)
    if spec['trainable']:
        ga=torch.autograd.grad(ka.sum(),th)[0];gb=torch.autograd.grad(kb.sum(),th)[0]
        np.testing.assert_allclose(ga,gb,atol=2e-6)
        ka,kb=ka.detach().numpy(),kb.detach().numpy()
        assert np.max(abs(gb.numpy()[:,3:]))<1e-8  # documented trailing-unitary cancellation
        assert np.max(abs(gb.numpy()[:,:3]))>1e-5
    np.testing.assert_allclose(ka,kb,atol=2e-6)


def test_raw_preprocessing_preserves_legacy_split():
    X,Z,y,t,*_=raw_split('wine',7)
    p=AnglePreprocessor(4,7).fit(X)
    a,b,c,d=load_dataset('wine',n_qubits=4,seed=7,max_train=40,max_test=20)
    np.testing.assert_allclose(p.transform(X),a,atol=1e-12)
    np.testing.assert_allclose(p.transform(Z),b,atol=1e-12)
    np.testing.assert_array_equal(y,c);np.testing.assert_array_equal(t,d)


@pytest.mark.parametrize('variant',['hard','soft','margin_weighted'])
def test_targets_correct_only_and_all_wrong_excluded(variant):
    d=np.array([[2.,1.,-3.],[-1.,-2.,-3.],[0.,0.,0.]])
    q,w=make_targets(d,np.ones(3),variant)
    np.testing.assert_allclose(q.sum(1),1)
    assert q[0,2]==0 and w.tolist()==[1,0,0]
    logits=torch.randn(3,3,requires_grad=True)
    imitation_loss(logits,q,w).backward()
    assert torch.count_nonzero(logits.grad[1:])==0
    if variant=='soft':np.testing.assert_allclose(q[0],[.5,.5,0])
    if variant=='margin_weighted':np.testing.assert_allclose(q[0],[2/3,1/3,0])


def test_all_wrong_batch_finite_zero_gradient():
    q,w=make_targets(-np.ones((4,3)),np.ones(4),'hard')
    l=torch.randn(4,3,requires_grad=True);loss=imitation_loss(l,q,w);loss.backward()
    assert loss.item()==0 and torch.count_nonzero(l.grad)==0


def test_fold_teacher_never_uses_held_labels():
    X,_,y,_,*_=raw_split('wine',0,max_train=20)
    fit=np.arange(14);held=np.arange(14,20)
    cfg={'n_qubits':4,'kernel_backend':'statevector','teacher_epochs':2,'teacher_lr':.05,'svm_C':1}
    d,meta=fit_fold_experts(X,y,fit,held,cfg,0)
    y2=y.copy();y2[held]*=-1
    d2,meta2=fit_fold_experts(X,y2,fit,held,cfg,0)
    np.testing.assert_allclose(d,d2,atol=0);assert meta==meta2
    X2=X.copy();X2[held]+=100
    _,m3=fit_fold_experts(X2,y,fit,held,cfg,0)
    assert meta['preprocessing']==m3['preprocessing'] and meta['theta']==m3['theta']


def test_statistics_zero_ties_holm_and_invalid():
    r=paired_comparison([.5]*5,[.5]*5,n_boot=100)
    assert r['p_ttest']==r['p_wilcoxon']==1 and r['cohens_dz']==0
    h=holm_correction({'a':.01,'b':.04,'c':.03})
    assert h['a']['p_adjusted']==.03 and h['b']['p_adjusted']==.06
    with pytest.raises(ValueError):paired_comparison([1,2],[1])
    with pytest.raises(ValueError):holm_correction([float('nan')])
    c=capture_rate([.5,.5],[.5,.5],[.5,.5],n_boot=100)
    assert c['estimate'] is None and c['ci95_lo'] is None


def test_outer_test_labels_do_not_change_training(monkeypatch):
    import experiments.expert_imitation as exp
    cfg={'dataset':'wine','n_qubits':4,'max_train':20,'max_test':8,'test_size':.3,
         'inner_folds':2,'teacher_epochs':2,'teacher_lr':.05,'router_epochs':3,'router_lr':.01,
         'svm_C':1.,'kernel_backend':'statevector','torch_threads':1,'variants':['hard','soft','margin_weighted']}
    original=exp.raw_split
    a=exp.run_one(cfg,0)
    def flip(*args,**kwargs):
        X,Z,y,yt,tr,te=original(*args,**kwargs)
        return X,Z,y,-yt,tr,te
    monkeypatch.setattr(exp,'raw_split',flip)
    b=exp.run_one(cfg,0)
    assert a['theta']==b['theta'] and a['oof_decisions']==b['oof_decisions']
    assert a['selected_single_index']==b['selected_single_index']
    for v in a['methods']:
        for field in ['gates_train','gates_test','losses','targets','target_weights','predictions','gate_state']:
            assert a['methods'][v][field]==b['methods'][v][field]


def test_nested_probe_outer_targets_do_not_affect_fit(monkeypatch):
    # Verify nested construction structurally: every inner fold indexes ONLY its outer-fit subset.
    from experiments.diagnostics_audited import nested_probe
    X,_,y,_,*_=raw_split('wine',0,max_train=20)
    cfg={'n_qubits':4,'kernel_backend':'statevector','teacher_epochs':1,'teacher_lr':.05,'svm_C':1.,
         'probe_outer_folds':2,'probe_inner_folds':2,'probes':['linear','mlp','majority'],
         'mlp_hidden':[4],'mlp_max_iter':10}
    r=nested_probe(X,y,cfg,0)
    held_seen=[]
    for f in r['folds']:
        fit=np.array(f['fit_indices']);held=f['held_indices'];held_seen+=held
        assert not set(fit)&set(held)
        for g in f['inner_folds_relative_to_fit']:
            assert not set(fit[g['fit_indices']])&set(held)
            assert not set(fit[g['held_indices']])&set(held)
    assert sorted(held_seen)==list(range(20))
