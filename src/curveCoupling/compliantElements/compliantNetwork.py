import numpy as np
from curveCoupling.utils.matrixOperations import ref
from typing import Tuple

class compliantNetworkEqs:
    def __init__(self, disp_constr: np.ndarray, force_constr: np.ndarray, disp_out: np.ndarray, force_out: np.ndarray):
        self.disp_constr = disp_constr
        self.force_constr = force_constr
        self.disp_out = disp_out
        self.force_out = force_out
    
    @classmethod
    def from_Matrices(cls, ConstraintMatrices: np.ndarray, OutputMatrices: np.ndarray) -> 'compliantNetworkEqs':
        return compliantNetworkEqs(*_constraintMatrices_to_force_disp(ConstraintMatrices), *_outputMatrices_to_force_disp(OutputMatrices))

    def getConstraintMatrices(self):
        return _force_disp_to_constraintMatrices(self.disp_constr, self.force_constr)
    
    def getOutputMatrices(self):
        return _force_disp_to_outputMatrices(self.disp_out, self.force_out)
        
    def invertProblem(self, solve_for_idx) -> 'compliantNetworkEqs':
        def invertCase(constr: np.ndarray, out: np.ndarray):
            top = np.hstack(([-1], out))
            bottom = np.hstack((np.zeros((constr.shape[0],1)), constr))
            total = np.vstack((top, bottom))
            total_swapped = total.copy()
            total_swapped[:, [0, solve_for_idx+1]] = total_swapped [:, [solve_for_idx+1,0]]
            total_swapped_ref = ref(total_swapped, tol=1e-6)

            assert total_swapped_ref[0,0] == 1.0, "The equations might not be linearly independent"
            assert np.all(total_swapped_ref[1:,0] == 0), "Something went wrong in Row Echelon Reduction"
            return total_swapped_ref[1:,1:], -total_swapped_ref[0,1:]

        new_disp_constr, new_disp_out = invertCase(self.disp_constr, self.disp_out)
        new_force_constr, new_force_out = invertCase(self.force_constr, self.force_out)
        return compliantNetworkEqs(new_disp_constr, new_force_constr, new_disp_out, new_force_out)

def _force_disp_to_constraintMatrices(disp_constr: np.ndarray, force_constr: np.ndarray) -> np.ndarray:
    """
    Generate ConstraintMatrices from separated force and displacement constraints.
    
    Parameters:
    - disp_constr (np.ndarray): Displacement constraint matrix.
    - force_constr (np.ndarray): Force constraint matrix.
    
    Returns:
    np.ndarray: ConstraintMatrices.
    """
    num_curves = disp_constr.shape[1]
    assert force_constr.shape[1] == num_curves, "Constraint arrays should represent the same number of elements."
    ConstraintMatrices = np.zeros((num_curves - 1, num_curves, 2))
    assert disp_constr.size > 0 or force_constr.size > 0, "At least one of the constraints should not be of size zero."
    numConstr = 0
    if disp_constr.size > 0:
        numConstr_disp = disp_constr.shape[0]
        ConstraintMatrices[:numConstr_disp, :, 0] = disp_constr
        numConstr += numConstr_disp
    if force_constr.size > 0:
        numConstr_force = force_constr.shape[0]
        ConstraintMatrices[-numConstr_force:, :, 1] = force_constr
        numConstr += numConstr_force    
    assert numConstr == num_curves - 1, "Requires N-1 constraints."

    return ConstraintMatrices

def _force_disp_to_outputMatrices(disp_out: np.ndarray, force_out: np.ndarray) -> np.ndarray:
    """
    Generate OutputMatrices from separated force and displacement outputs.
    
    Parameters:
    - disp_out (np.ndarray): Displacement output matrix.
    - force_out (np.ndarray): Force output matrix.
    
    Returns:
    np.ndarray: OutputMatrices.
    """
    assert disp_out.shape[0] == force_out.shape[0], "Output arrays should represent the same number of elements"
    num_curves = disp_out.shape[0]
    
    OutputMatrices = np.zeros((2, num_curves, 2))
    OutputMatrices[0, :, 0] = disp_out
    OutputMatrices[1, :, 1] = force_out

    return OutputMatrices

def _findCases(A: np.ndarray, index: int) -> np.ndarray:
        non_zero_idx = np.where(np.any(A[:, :, index] != 0.0, axis=1))[0]
        for i in non_zero_idx:
            mask = np.ones_like(A[i], dtype=bool) 
            mask[:, index] = False
            assert np.all(A[i][mask] == 0), "Mixed matrix, cannot decompose"

        return non_zero_idx

def _constraintMatrices_to_force_disp(ConstraintMatrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate separated force and displacement constraints and outputs from ConstraintMatrices and OutputMatrices.
    
    Parameters:
    - ConstraintMatrices (np.ndarray): Constraint matrices.
    
    Returns:
    Tuple[np.ndarray, np.ndarray]:
    - Displacement constraint matrix.
    - Force constraint matrix.
    """
    constr_disp_idx = _findCases(ConstraintMatrices, 0)
    disp_constr = ConstraintMatrices[constr_disp_idx, :, 0]
    constr_force_idx = _findCases(ConstraintMatrices, 1)
    force_constr = ConstraintMatrices[constr_force_idx, :, 1]

    return disp_constr, force_constr

def _outputMatrices_to_force_disp(OutputMatrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate separated force and displacement constraints and outputs from ConstraintMatrices and OutputMatrices.
    
    Parameters:
    - OutputMatrices (np.ndarray): Output matrices.
    
    Returns:
    Tuple[np.ndarray, np.ndarray]:
    - Displacement output matrix.
    - Force output matrix.
    """
    out_disp_idx = _findCases(OutputMatrices, 0)
    assert len(out_disp_idx) == 1, "Only one element in OutputMatrices should represent a displacement output"
    disp_out = OutputMatrices[out_disp_idx[0], :, 0]
    
    out_force_idx = _findCases(OutputMatrices, 1)
    assert len(out_force_idx) == 1, "Only one element in OutputMatrices should represent a force output"
    force_out = OutputMatrices[out_force_idx[0], :, 1]

    return disp_out, force_out

# if __name__ == "__main__":
#     from curveCoupling.compliantElements import generate_circuit_equations
#     edges = [
#         ('Start', 'End'),
#         ('Start', 'A'),
#         ('A', 'End'),
#     ]
#     eqs = generate_circuit_equations(edges)
#     eqs.invertProblem(0)

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.