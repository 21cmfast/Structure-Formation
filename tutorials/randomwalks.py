from matplotlib.pylab import seed
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from typing import Callable
from py21cmfast.wrapper.cfuncs import evaluate_sigma, get_growth_factor, return_uhmf_value, return_chmf_value
from py21cmfast import InputParameters
import attrs
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

@attrs.define(frozen=True)
class ST_barrier_params:
    #Jenkins defaults
    a: float = 0.73
    b: float = 0.34
    c: float = 0.81

PS_delta_c = 1.686
default_params = ST_barrier_params()
default_p21c_inputs = InputParameters(random_seed=1)

def setup_walk_steps(
    *,
    start_sigma: float | None = None,
    start_mass: float | None = None,
    min_mass: float = 1e8,
    max_mass: float = 1e16,
    num_steps: int = 100,
):
    """Set up the steps for the random walk.
    
    This function should not need to be altered during the tutorial.
    """
    if (start_sigma is None and start_mass is None) or (start_sigma is not None and start_mass is not None):
        raise ValueError("Exactly one of start_sigma or start_mass must be provided.")
    start_sigma_val = M_to_sigma(start_mass) if start_mass is not None else start_sigma
    start_mass_val = sigma_to_M(start_sigma) if start_sigma is not None else start_mass
    mass_steps = np.logspace(np.log10(max_mass), np.log10(min_mass), num_steps)
    mass_steps = mass_steps[mass_steps < start_mass_val]
    sigma_steps = M_to_sigma(mass_steps)
    sigma_steps = np.concatenate(([start_sigma_val],sigma_steps))
    mass_steps = np.concatenate(([start_mass_val],mass_steps))
    return sigma_steps, mass_steps

def barrier_PS(sigma: np.ndarray, growthf: np.ndarray) -> np.ndarray:
    """The Press-Schechter barrier function.
    
    The student should write this function during the tutorial.
    """
    return np.full((sigma.shape[0],growthf.shape[0]),PS_delta_c)/growthf[None,:]

def barrier_ST(sigma: np.ndarray, growthf: np.ndarray, params: ST_barrier_params = default_params) -> np.ndarray:
    """The Sheth-Tormen barrier function.

    The student should write this function during the tutorial,
    and may mess with the parameters to obtain different results.
    """
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
    
    sigma, _ = evaluate_sigma(inputs=default_p21c_inputs, masses=M)
    return sigma.squeeze()

def random_steps_sharpk(
    sigma_arr: np.ndarray,
    n_walks: int,
    start_delta: float = 0.0,
    seed: int = 12345,
):
    """
    Calculate the steps for the sharp-k filtered random walk.

    The student should write this function during the tutorial.
    """
    rng = np.random.default_rng(seed)
    start_points = np.full((n_walks, 1), start_delta)
    steps = np.concatenate(
        (
            start_points, #starting point
            rng.normal(
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
    """This may be implemented for an extended tutorial."""
    pass

def random_steps_gaussian(
    sigma_array: np.ndarray,
    start_delta: float = 0.0,
):
    """This may be implemented for an extended tutorial."""
    pass

def get_crossing_points(
    delta_arr: np.ndarray,
    barrier: np.ndarray,
):
    """
    Calculate halo masses given a random walk.

    Perhaps the student should write this function during the tutorial.
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
        delta: float = 0.0,
        halo_sigma: np.ndarray = None,
        halo_masses: np.ndarray = None,
        barrier: np.ndarray = None,
        output_file: str = None,
        hmf_type: str = 'PS',
        n_walks_plot: int = 10,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        color: str | None = None,
        label: str = "Random Walks",
        seed: int = 12345,
        plot_crossings: bool = False,
):
    """
    Plot the random walks and the barrier.

    This function should not need to be altered during the tutorial.
    """

    n_rows = 1 + (1 if halo_sigma is not None else 0) + (1 if halo_masses is not None else 0)
    height_ratios = [2.5,]
    fig_height = 4
    if halo_sigma is not None:
        height_ratios.insert(0,1)
        fig_height += 2
    if halo_masses is not None:
        height_ratios.append(1)
        fig_height += 2

    first_plot = (fig is None)
    if first_plot:
        fig, ax = plt.subplots(nrows=n_rows,ncols=1,figsize=(6,fig_height),layout='constrained',height_ratios=height_ratios)
        ax = np.atleast_1d(ax)
        fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0, wspace=0.0)

    walk_color = color if color is not None else 'r'
    line_color = color if color is not None else 'k'
    
    walk_ax = ax[(1 if halo_sigma is not None else 0)]

    #select random walks to plot
    rng = np.random.default_rng(seed=seed)
    sel = rng.choice(walks.shape[0], n_walks_plot, replace=False)

    #plot the random walks
    if barrier is not None and first_plot:
        walk_ax.plot(sigma_steps, barrier, color='k', linestyle='--', linewidth=2.5, label=f"{hmf_type} Barrier")
        
    walk_ax.plot(sigma_steps, walks[sel,:].T, color=walk_color, alpha=0.5, label=label)
    walk_ax.set_xlabel(r"$\sigma (M)$")
    walk_ax.set_ylabel(r"$\delta$")
    walk_ax.set_ylim(min(-1,delta-0.2),10)

    handles, labels = walk_ax.get_legend_handles_labels()
    labels,id = np.unique(labels, return_index=True)
    handles = [handles[i] for i in id]
    labels = [str(l) for l in labels]
    walk_ax.legend(loc='upper left', handles=handles, labels=labels, fontsize=8)
    if halo_sigma is not None and plot_crossings:
        crossings = np.argwhere(sigma_steps[None,None,:]==halo_sigma[sel,:,None]) #(walk,z,step)
        walk_ax.scatter(halo_sigma[sel][crossings[:,0]], barrier[crossings[:,-1]], color=walk_color, marker='x', s=50)

    #plot the sigma distribution
    if halo_sigma is not None:
        sigma_ax = ax[0]
        sigma_ax.sharex(walk_ax)
        #makes sure the values are not on the edges
        sigma_bins = np.concatenate(
            (
                sigma_steps[0:1] - np.diff(sigma_steps)[0]/2,
                sigma_steps[1:] - np.diff(sigma_steps)/2,
                sigma_steps[-2:-1] + np.diff(sigma_steps)[-1]/2
            )
        )
        sigma_hist,_ = np.histogram(
            halo_sigma,
            bins=sigma_bins,
        )
        sigma_hist = sigma_hist / np.sum(halo_sigma > 0)
        
        sigma_ax.step(sigma_steps, sigma_hist, color=line_color, linestyle='-', label="Sigma", linewidth=2.5)
        if plot_crossings:
            [sigma_ax.axvline(halo_sigma[s],color=walk_color) for s in sel]
        sigma_ax.tick_params(bottom=False,labelbottom=False)
        sigma_ax.set_ylabel("Fraction of Crossings")

    #plot the mass distribution
    if halo_masses is not None:
        inputs = default_p21c_inputs.evolve_input_structs(HMF=hmf_type)
        mass_dens = (
            inputs.cosmo_params.cosmo.critical_density(0).to("M_sun Mpc^-3").value
            * inputs.cosmo_params.cosmo.Om0
        )

        mass_ax = ax[(2 if halo_sigma is not None else 1)]
        #define bins to each contain one step increasing in mass
        mass_bins = np.concatenate(
            (
                mass_steps[0:1] - np.diff(mass_steps)[0]/2,
                mass_steps[1:] - np.diff(mass_steps)/2,
                mass_steps[-2:-1] + np.diff(mass_steps)[-1]/2
            )
        )
        dlnm = np.concatenate(
            (np.log(mass_bins[:-1]) - np.log(mass_bins[1:]),)
        )
        hm_hist,_ = np.histogram(
            halo_masses,
            bins=mass_bins[::-1],
        )
        hm_hist = hm_hist[::-1] #reverse to be in decreasing mass order
        # for step in mass_steps:
        #     print()
        dndlnm_fac = (
            (1 / walks.shape[0]) * #N_walk(M) --> F_mass(M)
            mass_dens * #F_mass(M) --> rho(M)
            (1/mass_steps) * #rho(M) --> n(M)
            (1/dlnm) #n(M) --> dn/dlnM
        )
        hm_hist = hm_hist * dndlnm_fac
        one_halo = dndlnm_fac # one halo in each bin
        
        if sigma_steps[0] == 0.0 and delta == 0.0:
            hmf = return_uhmf_value(
                mass_values=mass_steps[1:],
                redshift=redshift,
                inputs=inputs,
            ) * mass_dens
        else:
            hmf = return_chmf_value(
                mass_values=mass_steps[1:],
                redshift=redshift,
                inputs=inputs,
                delta_values=np.array([delta]),
                condmass_values=np.array([mass_steps[0]]),
            ).squeeze() * mass_dens
        
        hmf = np.concatenate(([0],hmf)) #add zero at high mass end where no halos possible

        #although the mass bins contain the steps, the masses are exactly at the steps
        mass_ax.step(mass_steps, hm_hist, color=line_color, linestyle='-', label="Halo Mass", linewidth=2.5)
        mass_ax.plot(mass_steps, hmf, color=line_color, linestyle=':', label=hmf_type+' HMF')
        mass_ax.set_xscale('log')
        mass_ax.set_yscale('log')
        mass_ax.set_ylim(1e-7,1e1)
        mass_ax.set_xlim(1e8,2e13)
        mass_ax.set_ylabel("dn/dlnM [Mpc-3]")
        mass_ax.set_xlabel("Mass [M_sun]")
        if plot_crossings:
            [mass_ax.axvline(halo_masses[s],color=walk_color) for s in sel]
        #do this on first plot only
        if first_plot:
            mass_ax.plot(mass_steps, one_halo,color='k',linestyle='--',label='sampling limit')
            l_handles = [Line2D([0], [0], color='k', linestyle='-', label='Halo Mass'),
                        Line2D([0], [0], color='k', linestyle=':', label=hmf_type+' HMF'),
                        Line2D([0], [0], color='k', linestyle='--', label='Sampling Limit')]
            mass_ax.legend(loc='lower left', handles=l_handles, fontsize=8)

    if output_file:
        plt.savefig(output_file)
    return fig,ax