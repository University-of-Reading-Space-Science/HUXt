"""
Wrapper around pyro's single_step that allows custom boundary conditions.

This extracts pyro's time-stepping but allows us to prescribe density,
velocity, and temperature as functions of radius at the boundaries.
"""
import numpy as np
from pyro.pyro_sim import Pyro
from pyro.util import msg

# Physical constants (CGS)
PROTON_MASS = 1.67262192e-24  # grams
BOLTZMANN = 1.380649e-16  # erg/K


class PyroCustomBCWrapper:
    """
    Wrapper around pyro that allows custom boundary conditions to be 
    applied at each timestep before calling single_step.
    """
    
    def __init__(self, pyro_obj, gamma=5.0/3.0):
        """
        Initialize wrapper.
        
        Parameters
        ----------
        pyro_obj : Pyro
            An initialized Pyro simulation object
        gamma : float
            Adiabatic index
        """
        self.pyro = pyro_obj
        self.sim = pyro_obj.get_sim()
        self.gamma = gamma
        self.custom_bc_func = None
        
    def set_custom_bc(self, bc_func):
        """
        Set a custom boundary condition function.
        
        Parameters
        ----------
        bc_func : callable
            Function with signature: bc_func(r, t) -> (rho, v, T)
            where r is radius array, t is current time
            Should return density, velocity, temperature arrays
        """
        self.custom_bc_func = bc_func
        
    def apply_custom_bc(self):
        """
        Apply custom boundary conditions to ghost cells.
        
        This is called AFTER fill_BC_all() to override the standard BCs
        with user-prescribed values.
        
        For inflow boundaries, ghost cells should represent the state
        AT the boundary (xmin or xmax), not at the ghost cell centers.
        """
        if self.custom_bc_func is None:
            return
            
        myd = self.sim.cc_data
        myg = myd.grid
        
        # Get current time (strip units if present)
        t = myd.t
        if hasattr(t, 'value'):
            t = t.value
        
        # Get variables
        dens = myd.get_var("density")
        xmom = myd.get_var("x-momentum")
        ymom = myd.get_var("y-momentum")
        ener = myd.get_var("energy")
        
        # Apply to lower x boundary (inner radial boundary)
        # All ghost cells should have the state AT r = xmin (the boundary)
        if myg.ilo > 0:
            # Create array of boundary radii (one per angular cell)
            r_boundary = np.full(myg.ny, myg.xmin)
            rho_bc, v_bc, T_bc = self.custom_bc_func(r_boundary, t)
            
            # Set all inner ghost cells to the boundary state
            for i in range(myg.ilo):
                # Set conserved variables
                dens[i, :] = rho_bc
                xmom[i, :] = rho_bc * v_bc
                ymom[i, :] = 0.0  # No angular velocity
                
                # Compute energy
                P_bc = rho_bc * BOLTZMANN * T_bc / PROTON_MASS
                e_int = P_bc / (rho_bc * (self.gamma - 1.0))
                e_kin = 0.5 * v_bc**2
                ener[i, :] = rho_bc * (e_int + e_kin)
        
        # Apply to upper x boundary (outer radial boundary)
        # All ghost cells should have the state AT r = xmax (the boundary)
        if myg.ihi < myg.nx + 2*myg.ng - 1:
            # Create array of boundary radii
            r_boundary = np.full(myg.ny, myg.xmax)
            rho_bc, v_bc, T_bc = self.custom_bc_func(r_boundary, t)
            
            # Set all outer ghost cells to the boundary state
            for i in range(myg.ihi + 1, myg.nx + 2*myg.ng):
                # Set conserved variables
                dens[i, :] = rho_bc
                xmom[i, :] = rho_bc * v_bc
                ymom[i, :] = 0.0
                
                # Compute energy
                P_bc = rho_bc * BOLTZMANN * T_bc / PROTON_MASS
                e_int = P_bc / (rho_bc * (self.gamma - 1.0))
                e_kin = 0.5 * v_bc**2
                ener[i, :] = rho_bc * (e_int + e_kin)
    
    def single_step_with_custom_bc(self):
        """
        Perform a single timestep with custom boundary conditions.
        
        This mimics pyro's single_step but applies custom BCs.
        
        Returns
        -------
        dt : float
            The timestep used
        """
        # Fill standard boundary conditions first
        self.sim.cc_data.fill_BC_all()
        
        # Override with custom BCs
        self.apply_custom_bc()
        
        # Get timestep
        self.sim.compute_timestep()
        dt = self.sim.dt
        
        # Strip units if present
        if hasattr(dt, 'value'):
            dt = dt.value
        
        # Evolve
        self.sim.evolve()
        
        return dt
    
    def get_state(self):
        """
        Extract current state.
        
        Returns
        -------
        dict with keys:
            t : float
                Current time
            r : ndarray
                Radial coordinates (1D)
            rho : ndarray
                Density (averaged over angular direction)
            v : ndarray
                Radial velocity (averaged)
            T : ndarray
                Temperature (averaged)
            P : ndarray
                Pressure (averaged)
        """
        myd = self.sim.cc_data
        myg = myd.grid
        
        # Get coordinates (interior cells only)
        r = myg.x[myg.ilo:myg.ihi+1]
        
        # Get variables
        rho_array = myd.get_var("density").v()
        xmom_array = myd.get_var("x-momentum").v()
        ener_array = myd.get_var("energy").v()
        
        # Average over angular direction
        rho = np.mean(rho_array, axis=1)
        mom = np.mean(xmom_array, axis=1)
        E = np.mean(ener_array, axis=1)
        
        # Compute primitives
        v = mom / rho
        ke = 0.5 * rho * v**2
        P = (self.gamma - 1.0) * (E - ke)
        # Temperature from ideal gas law: P = n*k*T = (rho/m_p)*k*T
        # So T = P*m_p/(rho*k)
        T = P * PROTON_MASS / (rho * BOLTZMANN)
        # Number density from mass density
        n = rho / PROTON_MASS
        
        # Strip units from time if present (pyro may return astropy Quantity)
        t_value = myd.t
        if hasattr(t_value, 'value'):
            t_value = t_value.value
        
        return {
            't': t_value,
            'r': r,
            'rho': rho,
            'v': v,
            'T': T,
            'P': P,
            'n': n
        }
    
    def finished(self):
        """Check if simulation is finished."""
        return self.sim.finished()


# Example usage function
def example_solar_wind_with_fixed_bc():
    """
    Example: Solar wind with fixed inner boundary conditions.
    """
    from pyro.pyro_sim import Pyro
    
    # Constants
    AU = 1.496e13  # cm
    KM_TO_CM = 1e5
    
    # Domain
    r_inner = 0.1 * AU
    r_outer = 1.0 * AU
    nr = 200
    
    # Boundary values
    v_inner = 400.0 * KM_TO_CM  # cm/s
    n_inner = 200.0  # protons/cc
    T_inner = 1.0e6  # K
    rho_inner = n_inner * PROTON_MASS
    
    gamma = 5.0 / 3.0
    
    # Create pyro simulation
    pyro = Pyro("compressible")
    pyro.initialize_problem(
        problem_name="sedov",
        inputs_dict={
            "mesh.nx": nr,
            "mesh.ny": 1,
            "mesh.xmin": r_inner,
            "mesh.xmax": r_outer,
            "mesh.ymin": 0.0,
            "mesh.ymax": np.pi,
            "driver.tmax": 86400.0,  # 1 day
            "driver.max_steps": 100000,
            "driver.cfl": 0.3,
            "compressible.riemann": "CGF",
            "eos.gamma": gamma,
            "mesh.grid_type": "SphericalPolar",
            "mesh.xlboundary": "outflow",
            "mesh.xrboundary": "outflow",
        }
    )
    
    # Initialize with wind profile
    myd = pyro.sim.cc_data
    myg = myd.grid
    
    dens = myd.get_var("density")
    xmom = myd.get_var("x-momentum")
    ymom = myd.get_var("y-momentum")
    ener = myd.get_var("energy")
    
    r = myg.x2d
    v_rad = v_inner * np.ones_like(r)
    rho_rad = rho_inner * (r_inner / r)**2
    T_rad = T_inner * (r_inner / r)**0.7
    P_rad = rho_rad * BOLTZMANN * T_rad / PROTON_MASS
    
    dens[:, :] = rho_rad
    xmom[:, :] = rho_rad * v_rad
    ymom[:, :] = 0.0
    
    e_int = P_rad / (rho_rad * (gamma - 1.0))
    e_kin = 0.5 * v_rad**2
    ener[:, :] = rho_rad * (e_int + e_kin)
    
    # Create wrapper
    wrapper = PyroCustomBCWrapper(pyro, gamma=gamma)
    
    # Define custom BC function for inner boundary
    def solar_wind_bc(r, t):
        """
        Prescribe inner boundary: fixed v, rho, T
        Outer boundary: extrapolation (handled by pyro's outflow)
        """
        # For simplicity, return the wind profile
        # In practice, you'd check if r < some threshold
        rho = rho_inner * (r_inner / r)**2
        v = v_inner * np.ones_like(r)
        T = T_inner * (r_inner / r)**0.7
        return rho, v, T
    
    wrapper.set_custom_bc(solar_wind_bc)
    
    # Run a few steps
    print("Running with custom BC...")
    for step in range(20):
        dt = wrapper.single_step_with_custom_bc()
        
        if step % 5 == 0:
            state = wrapper.get_state()
            print(f"Step {step}, t={state['t']:.3f} s, dt={dt:.3f} s")
            print(f"  v range: [{state['v'].min()/KM_TO_CM:.1f}, {state['v'].max()/KM_TO_CM:.1f}] km/s")
            print(f"  rho range: [{state['rho'].min():.3e}, {state['rho'].max():.3e}]")
        
        if wrapper.finished():
            break
    
    print("\nDone!")
    return wrapper


if __name__ == "__main__":
    example_solar_wind_with_fixed_bc()
