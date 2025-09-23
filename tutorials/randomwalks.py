import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from typing import Callable
from py21cmfast.wrapper.cfuncs import evaluate_sigma, get_growth_factor, return_uhmf_value
from py21cmfast import InputParameters
import attrs
from tqdm import tqdm

from matplotlib import pyplot as plt

@attrs.define(frozen=True)
class ST_barrier_params:
    #Jenkins defaults
    a: float = 0.73
    b: float = 0.34
    c: float = 0.81

PS_delta_c = 1.686
default_params = ST_barrier_params()
default_p21c_inputs = InputParameters(random_seed=1)

def barrier_PS(sigma: np.ndarray, growthf: np.ndarray) -> np.ndarray:
    return np.full((sigma.shape[0],growthf.shape[0]),PS_delta_c)/growthf[None,:]

def barrier_ST(sigma: np.ndarray, growthf: np.ndarray, params: ST_barrier_params = default_params) -> float:
    delta = PS_delta_c / growthf[None,:]
    return np.sqrt(params.a) * delta * (
        1. + params.b * pow(sigma[:,None] * sigma[:,None] /
        (params.a * delta * delta), params.c)
    )

def sigma_to_M(sigma: np.ndarray | float) -> np.ndarray:
    if isinstance(sigma, float):
        sigma = np.array([sigma])
    result = np.zeros_like(sigma)
    for i,sig in enumerate(sigma):
        if sig <= 0:
            result[i] = np.inf
            continue
        func = lambda x: M_to_sigma(x) - sig
        # print(f"Finding mass for sigma {sig} between {1e7} {M_to_sigma(1e7)} and {1e16} {M_to_sigma(1e16)}")
        sol = root_scalar(func,bracket=(1e7, 1e20))
        if not sol.converged:
            raise ValueError("Root finding did not converge.")
        result[i] = sol.root
    return result.squeeze()

def M_to_sigma(M: np.ndarray | float) -> np.ndarray | float:
    if isinstance(M, float):
        M = np.array([M])
    
    sigma, dsigma = evaluate_sigma(inputs=default_p21c_inputs, masses=M)
    return sigma.squeeze()

def random_steps_sharpk(
    sigma_arr: np.ndarray,
    n_walks: int,
    start_delta: float = 0.0,
):
    """
    Calculate the steps for the sharp-k filtered random walk.
    """
    start_points = np.full((n_walks, 1), start_delta)
    steps = np.concatenate(
        (
            start_points, #starting point
            np.random.normal(
                loc=0.,
                scale=np.sqrt(np.diff(sigma_arr**2)),
                size=(n_walks, len(sigma_arr)-1),
            ),
        ),
        axis=1,
    )
    return np.cumsum(steps,axis=1)

def random_steps_tophat(
    sigma_array: np.ndarray,
    start_delta: float = 0.0,
):
    pass

def random_steps_gaussian(
    sigma_array: np.ndarray,
    start_delta: float = 0.0,
):
    pass

def get_crossing_points(
    delta_arr: np.ndarray,
    barrier: np.ndarray,
):
    """
    Calculate halo masses given a random walk.
    """
    # Calculate the mass from the delta array
    #(walk,sigma,z)
    halo_bool = np.any(delta_arr[:,:,None] > barrier[None,:,:], axis=1)
    crossing_points = np.where(
        halo_bool,
        np.argmax(delta_arr[:,:,None] > barrier[None,:,:], axis=1),
        -1,
    )
    return crossing_points

def plot_walks(
        *,
        redshift: float,
        walks: np.ndarray,
        sigma_steps: np.ndarray,
        mass_steps: np.ndarray,
        halo_sigma: np.ndarray = None,
        halo_masses: np.ndarray = None,
        barrier: np.ndarray = None,
        output_file: str = None,
        hmf_type: str = 'PS',
        n_walks_plot: int = 10,
):
    """
    Plot the random walks and the barrier.
    """

    n_rows = 1 + (1 if halo_sigma is not None else 0) + (1 if halo_masses is not None else 0)
    height_ratios = [2.5,]
    if halo_sigma is not None:
        height_ratios.insert(0,1)
    if halo_masses is not None:
        height_ratios.append(1)
    fig, ax = plt.subplots(nrows=n_rows,ncols=1,figsize=(6,6),layout='constrained',height_ratios=height_ratios)
    ax = np.atleast_1d(ax)
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0, wspace=0.0)
    
    walk_ax = ax[(1 if halo_sigma is not None else 0)]
    sel = np.random.choice(walks.shape[0], n_walks_plot, replace=False)
    #plot the random walks
    walk_ax.plot(sigma_steps, walks[sel,:].T,'r-', alpha=0.5)
    if barrier is not None:
        walk_ax.plot(sigma_steps, barrier, 'k:', label="Barrier", linewidth=2.5)
    walk_ax.set_xlabel("Sigma")
    walk_ax.set_ylabel("Delta")
    walk_ax.set_ylim(-1,10)

    #plot the sigma distribution
    if halo_sigma is not None:
        sigma_ax = ax[0]
        sigma_ax.sharex(walk_ax)
        #makes sure the values are not on the edges
        sigma_bins = np.append(sigma_steps[1:] - np.diff(sigma_steps)/2,
                               sigma_steps[-1] + np.diff(sigma_steps)[-1]/2)
        sigma_hist,_ = np.histogram(
            halo_sigma,
            bins=sigma_bins,
        )
        sigma_hist = sigma_hist / sigma_hist.max()
        
        sigma_ax.step(sigma_steps[1:], sigma_hist, 'k-', label="Sigma", linewidth=2.5)
        [sigma_ax.axvline(halo_sigma[s],color='r') for s in sel]
        sigma_ax.tick_params(bottom=False,labelbottom=False)
        sigma_ax.set_ylabel("N/Nmax")
    

    #plot the mass distribution
    if halo_masses is not None:
        mass_ax = ax[(2 if halo_sigma is not None else 1)]
        mass_bins = np.append(mass_steps[1:] - np.diff(mass_steps)/2,
                               mass_steps[-1] + np.diff(mass_steps)[-1]/2)
        hm_hist,_ = np.histogram(
            halo_masses,
            bins=mass_bins[::-1],
        )
        hm_hist = hm_hist[::-1] / hm_hist.max()
        
        hmf = return_uhmf_value(
            mass_values=mass_steps[1:],
            redshift=redshift,
            inputs=default_p21c_inputs.evolve_input_structs(HMF=hmf_type),
        ) * mass_steps[1:]
        hmf = hmf/hmf.max()

        #although the mass bins contain the steps, the masses are exactly at the steps
        mass_ax.step(mass_steps[1:], hm_hist, 'k-', label="Halo Mass", linewidth=2.5)
        mass_ax.plot(mass_steps[1:],hmf,'k:',label=hmf_type+' HMF')
        mass_ax.set_xscale('log')
        mass_ax.set_yscale('log')
        mass_ax.set_ylim(1e-3,1e0)
        mass_ax.set_ylabel("N/Nmax")
        mass_ax.set_xlabel("Mass")
        mass_ax.legend()
        [mass_ax.axvline(halo_masses[s],color='r') for s in sel]

    if output_file:
        plt.savefig(output_file)
    return fig,ax

if __name__ == "__main__":
    min_mass = 1e8
    max_mass = 1e15
    num_steps = 100
    start_sigma = 0.
    start_delta = 0.
    redshift = 6.0
    growthf = np.array([get_growth_factor(redshift=redshift, inputs=default_p21c_inputs)])
    n_walks = 100000

    mass_steps = np.logspace(np.log10(max_mass), np.log10(min_mass), num_steps)
    sigma_steps = M_to_sigma(mass_steps)
    sigma_steps = np.concatenate(([start_sigma],sigma_steps))
    mass_steps = np.concatenate(([sigma_to_M(start_sigma)],mass_steps))
    barrier_steps = barrier_PS(sigma_steps, growthf)
    
    walks = random_steps_sharpk(sigma_steps,n_walks,start_delta)
    crossing_points = get_crossing_points(
        delta_arr=walks,
        barrier=barrier_steps,
    )
    halo_masses = np.where(crossing_points==-1,np.nan,mass_steps[crossing_points])
    halo_sigma = np.where(crossing_points==-1,np.nan,sigma_steps[crossing_points])

    fig,ax = plot_walks(
        walks=walks,
        sigma_steps=sigma_steps,
        mass_steps=mass_steps[1:],
        halo_sigma=halo_sigma,
        halo_masses=halo_masses,
        barrier=barrier_steps,
        output_file="random_walks.png",
    )