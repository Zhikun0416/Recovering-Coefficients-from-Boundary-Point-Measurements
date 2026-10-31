"""Geometry-calibrated A-optimal Navier inputs for both geometries."""

import numpy as np

from .backend import MODELS, Problem, make_design, parameters, reference_boundary


def response_calibration(problem):
    """Boundary-L2-normalized lifting norms evaluated at the zero model."""
    gram=reference_boundary(problem.case)[1]
    size=len(gram)
    zero=np.zeros(size)
    norms=[]
    eigenvalues, vectors = np.linalg.eigh(gram)
    orthonormal = vectors[:, eigenvalues > 1e-12]/np.sqrt(eigenvalues[eigenvalues > 1e-12])
    for component in ('f','g'):
        total=0.
        # Trace in an arclength-orthonormal basis, including off-diagonal Gram entries.
        for unit in orthonormal.T:
            row=dict(f=(unit if component=='f' else zero).tolist(),
                     g=(unit if component=='g' else zero).tolist())
            _,field,_=problem.solve(np.zeros(len(parameters(problem.model)[0])),problem.prepare([row])[0])
            total+=np.sum(field**2*problem.weights)
        norms.append(total)
    return float(np.sqrt(norms[0]/norms[1]))


def candidates(case, ex, seed, gain, count=128):
    amplitude=.4 if ex==1 else .6
    original=make_design(case,4,4*count,seed,amplitude=amplitude)
    gram=reference_boundary(case)[1]
    spectral_weights=np.tile([1.,1.,.5,1/3],2)
    for name,rows in original.items():
        for row in rows:
            f=np.array(row['f'])*spectral_weights
            g=np.array(row['g'])*spectral_weights
            divisor=np.sqrt(2) if name=='dual' else 1.
            if np.any(f): f*=amplitude*abs(row['scale'])/(np.sqrt(f@gram@f)*divisor)
            if np.any(g): g*=gain*amplitude*abs(row['scale'])/(np.sqrt(g@gram@g)*divisor)
            row.update(f=f.tolist(),g=g.tolist(),
                       weighted_energy=float(f@gram@f+g@gram@g/gain**2),
                       f_energy_fraction=1. if name=='f_only' else 0. if name=='g_only' else .5)
    if ex==4:
        expanded=[]
        for share in (.1,.25,.5,.75,.9):
            for polarity in (1.,-1.):
                for row in original['dual']:
                    expanded.append(dict(row,f=(np.array(row['f'])*np.sqrt(2*share)).tolist(),
                                         g=(np.array(row['g'])*np.sqrt(2*(1-share))*polarity).tolist(),
                                         f_energy_fraction=share,g_polarity=polarity))
        original['dual']=expanded
        return original
    combined=[]
    for direction in range(count):
        name=('f_only','g_only','dual','dual')[direction%4]
        indices=(2,3) if ex==1 else range(4)
        combined.extend(original[name][4*direction+k] for k in indices)
    return {'all':combined}


def information_matrices(case, rows, ex):
    """Use the fixed initial model; EX3 balances both candidate nonlinear orders."""
    block_size=2 if ex==1 else 4
    models=('quadratic','cubic') if ex==3 else (MODELS[ex],)
    pieces=[]
    nominal=[]
    for model in models:
        problem=Problem(case,model,32)
        initial=parameters(model)[1]
        inputs=problem.prepare(rows)
        if model=='radial':
            jac=np.column_stack([(problem.measure(initial+np.eye(len(initial))[j]*1e-4,inputs)-problem.measure(initial-np.eye(len(initial))[j]*1e-4,inputs))/2e-4 for j in range(len(initial))])
        else:
            _,jac=problem.measure(initial,inputs,gradient=True)
        blocks=(jac*np.abs(initial)).reshape(-1,block_size,len(initial))
        pieces.append(np.einsum('nki,nkj->nij',blocks,blocks))
        nominal.append(dict(model=model,parameters=initial.tolist()))
    size=sum(piece.shape[1] for piece in pieces)
    information=np.zeros((len(pieces[0]),size,size))
    offset=0
    for piece in pieces:
        end=offset+piece.shape[1]
        information[:,offset:end,offset:end]=piece
        offset=end
    return information,nominal


def choose(information,count,kinds=None):
    dimension=information.shape[-1]
    ridge=float(np.trace(information.sum(axis=0))*1e-10/dimension)
    current=np.eye(dimension)*ridge
    selected=[]
    quota={'f_only':count//4,'g_only':count//4,'dual':count//2} if kinds is not None else None
    def scores(matrices):
        return np.sum(1/np.maximum(np.linalg.eigvalsh(matrices),1e-30),axis=-1)
    for _ in range(count):
        score=scores(current[None]+information)
        score[selected]=np.inf
        if quota:
            for kind,budget in quota.items():
                if sum(kinds[i]==kind for i in selected)>=budget:
                    score[np.array(kinds)==kind]=np.inf
        index=int(np.argmin(score))
        assert np.isfinite(score[index])
        selected.append(index)
        current+=information[index]
    exchanges=0
    for iteration in range(5):
        changed=False
        for position,old in enumerate(selected):
            rest=current-information[old]
            score=scores(rest[None]+information)
            score[[i for i in selected if i!=old]]=np.inf
            if kinds is not None:score[np.array(kinds)!=kinds[old]]=np.inf
            new=int(np.argmin(score))
            if score[new]<score[old]*(1-1e-10):
                selected[position]=new
                current=rest+information[new]
                changed=True;exchanges+=1
        if not changed:break
    return selected,dict(ridge=ridge,exchanges=exchanges,criterion=float(scores(current[None])[0]))


def make_optimal_design(case,ex,n,seed):
    block_size=2 if ex==1 else 4
    gain=response_calibration(Problem(case,'quadratic',32))
    pools=candidates(case,ex,seed,gain)
    result={}
    report=dict(method='A-optimal design at the fixed initial model, before generating observations or noise',
                seed=seed,geometry='unit disk' if case==1 else 'analytic rounded rectangle',g_scale=gain,
                calibration='ratio of Frobenius norms of boundary-L2-normalized biharmonic lifting maps',
                spectral_decay='mode m has coefficient weight 1/max(1,m)',
                energy='arclength mean of f^2+g^2/g_scale^2',strategies={})
    for name,rows in pools.items():
        information,nominal=information_matrices(case,rows,ex)
        kinds=[rows[i]['kind'] for i in range(0,len(rows),block_size)] if ex!=4 else None
        selected,diagnostics=choose(information,n//block_size,kinds)
        chosen=[]
        for direction,index in enumerate(selected):
            for k in range(block_size):
                chosen.append(dict(rows[block_size*index+k],direction=direction,candidate_index=index))
        result[name]=chosen
        report['strategies'][name]=dict(nominal=nominal,candidate_directions=len(information),
                                      selected_indices=selected,**diagnostics)
        print(f'  Case{case} design {name}: {len(information)} candidates -> {len(selected)} directions',flush=True)
    return (result if ex==4 else result['all']),report
