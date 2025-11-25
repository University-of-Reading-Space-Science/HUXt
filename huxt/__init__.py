from huxt.huxt import HUXt, ConeCME

# Expose compressible solvers for advanced usage
from huxt.compressible_solvers import (
    CompressibleSolver,
    create_solver,
    benchmark_solvers,
    list_available_methods,
)

# Backward compatibility: CGFSolver is also available
from huxt.cgf_solver import CGFSolver
