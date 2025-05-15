# Structure-Formation
Structure formation tutorials using 21cmFAST.  These are intended as hands-on excercises that acompany the course Structure formation in the early Universe.  The course textbook (in prep) is included in the repo.

Suggestions for tutorials:

1) Make and visualize initial conditions (density + velocities).  Change seeds. Change cosmology: PS, transfer function cutt-off; non-Gaussianity (maybe?)
Tools: power spectra calculator; 2d slice visualization..?

2) Perturb and visualize ICs Lagrangian → Eularian.  Just scale amplitudes (Eularian PT) vs ZA vs 2LPT.  plot PS and slices
Tools: power spectra calculator; 2d slice visualization

3) Halo finding.  Run DexM halo finder.  Alternate barrier from PS and ST.  Compute HMFs.  Visualize halos on top of the IC fields in Lagrangian space.  Perturb both halos and density to different redshifts. Visualize slices and compute the HMFs.  Maybe compare DexM halo field with a M_200 halo field.
Tools: 2d slice visualization of field + sources; 

4) Compute the halo power spectrum and compare it to Lagrangian linear matter power spectrum and Eularian linear matter power spectrum (from previous).  Compute bias.  Compare with halo model (maybe?)

5) Put galaxies inside halos by sampling P(M_star | M_halo) and P(SFR | M_star).  Start with no scatter.  Visualize galaxies on top of the matter field, compute UV LFs, compute PS/bias as a function of M_uv cut. Vary the relations and add scatter, and see how the UV LFs and bias change. Show the effects of RSDs over the observables?

6) Compute EoR.  Visualize galaxies on top of the EoR field.  Vary f_esc(M) and repeat.

7) Add Lyman alpha emission to galaxies by sampling conditional relations.  Show emergent vs observed luminosities.  Compare for a specific galaxy in the center vs edge of a bubble.  Compute statistics: Lya LFs at different xHI, and clustering.

8) Compute and visualize the 21-cm signal, lightcone and coevals.  Compute global evolution and 21cm PS (maybe as a 2D image).  Change main parameters and recompute. Show the effects of RSDs over the observables.

9) Compute and visualize the kSZ signal.

