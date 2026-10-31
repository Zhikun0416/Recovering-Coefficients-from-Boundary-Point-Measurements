"""The two manuscript overviews: q and F(x,u), each with three columns."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np

from .backend import Problem, nonlinear
from .protocol import DATA_LEVEL

def probe(case):
    f, g = [.08,.22,0.,.04,0.,0.,.08,0.], [0.,0.,-.18,0.,0.,.30,0.,0.]
    description = "f=0.08+0.22cos(theta)+0.08sin(2theta)+0.04cos(3theta); g=0.30sin(theta)-0.18cos(2theta)"
    return [dict(f=f, g=g)], description


def geometry(problem):
    nr, nt = problem.disk.nr, problem.disk.nt
    r = np.linspace(0,1,nr+1)[:,None]
    theta = (np.arange(nt+1)-.5)*2*np.pi/nt
    if problem.case == 2:
        from .rounded_rectangle import conformal_map
        z = conformal_map(r*np.exp(1j*theta))
        return z.real, z.imag, "flat"
    return r*np.cos(theta), r*np.sin(theta), "flat"


def draw_row(fig,axes,problem,pair,limits,label):
    a,b = pair
    lo,hi,err = limits
    xx,yy,shading = geometry(problem)
    maps=[]
    for ax,z,title,cmap,norm in zip(axes,(a,b,abs(b-a)),("Truth","Reconstruction","Absolute error"),
                                   ("viridis","viridis","magma"),(Normalize(lo,hi),Normalize(lo,hi),Normalize(0,err or 1e-16))):
        maps.append(ax.pcolormesh(xx,yy,z,shading=shading,cmap=cmap,norm=norm,rasterized=True))
        ax.set(xlabel="$x_1$",ylabel="$x_2$",aspect="equal")
        if problem.case==1:
            ax.add_patch(plt.Circle((0,0),1,fill=False,color=".3",lw=.6))
            ax.set(xlim=(-1.04,1.04),ylim=(-1.04,1.04),xticks=[-1,0,1],yticks=[-1,0,1])
        else:
            ax.set(xlim=(0,2),ylim=(0,1),xticks=[0,.5,1,1.5,2],yticks=[0,.5,1])
        ax.set_title(f"{label}\n{title}",fontsize=14)
    for mappable,targets in ((maps[0],list(axes[:2])),(maps[2],axes[2])):
        bar=fig.colorbar(mappable,ax=targets,fraction=.04,pad=.035,shrink=.8)
        maximum=max(abs(mappable.norm.vmin),abs(mappable.norm.vmax))
        exponent=int(np.floor(np.log10(maximum))) if maximum else 0
        if maximum and (maximum<.01 or maximum>=1e4):
            bar.formatter=FuncFormatter(lambda value,_,e=exponent:f"{value/10.**e:.4f}")
            bar.ax.set_title(rf"$\times10^{{{exponent}}}$",fontsize=12,pad=8)
        else:
            bar.formatter=FuncFormatter(lambda value,_:f"{value:.4f}")
        bar.update_ticks()


def _fields(case, recovered):
    problem = Problem(case, "quadratic", DATA_LEVEL)
    design, _ = probe(case)
    boundary = problem.prepare(design)[0]
    from .backend import parameters
    truth = parameters("quadratic")[0]
    _, true_state, _ = problem.solve(truth, boundary)
    _, recovered_state, _ = problem.solve(recovered, boundary)
    qt, at, bt = problem.fields(truth)
    qr, ar, br = problem.fields(recovered)
    return problem, {
        "q": (qt, qr),
        "F": (nonlinear(problem.model, at, bt, true_state),
              nonlinear(problem.model, ar, br, recovered_state)),
    }


def plot_overviews(summaries, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "pdf.fonttype": 42})
    data = {case: _fields(case, summaries[(case, 4)]["strategies"]["dual"]["cross_finer"]["recovered"])
            for case in (1, 2)}
    limits = {}
    for name in ("q", "F"):
        pairs = [data[case][1][name] for case in (1, 2)]
        limits[name] = (float(min(min(a.min(), b.min()) for a, b in pairs)),
                        float(max(max(a.max(), b.max()) for a, b in pairs)))
    for case, (problem, fields) in data.items():
        fig, axes = plt.subplots(2, 3, figsize=(12, 8 if case == 1 else 6.6), layout="constrained")
        for row, name, label in zip(axes, ("q", "F"), ("$q(x)$", "$F(x,u(x))$")):
            pair = fields[name]
            error = float(np.abs(pair[1] - pair[0]).max())
            draw_row(fig, row, problem, pair, (*limits[name], error), label)
        domain = "unit disk" if case == 1 else "rounded rectangle"
        fig.suptitle(f"Case {case}: {domain}, simultaneous $f,g$ inputs (EX4)", fontsize=16)
        fig.savefig(output / f"case{case}_ex4_dual.pdf", dpi=200,
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
