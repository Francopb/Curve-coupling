import numpy as np
from typing import List
from curveCoupling.utils.matrixOperations import ref

class separableEqs:
    def __init__(self, constraintMatrices_lst: List[np.ndarray], outputVectors_lst: List[np.ndarray]):
        numCurves = constraintMatrices_lst[0].shape[1]
        nDims = len(constraintMatrices_lst)
        assert all([c.shape[1] == numCurves for c in constraintMatrices_lst]), "All constraintMatrices need to have the same number of columns"
        assert all([c.size == numCurves for c in outputVectors_lst]), "All outputVectors need to have the same length as columns in constraintMatrices"
        assert len(outputVectors_lst) == nDims, "Required as many outputVectors as dimensions"
        assert sum([c.shape[0] for c in constraintMatrices_lst]) == numCurves-1, "Required N-1 constraints (N is number of curves)"
        self.constraintMatrices_lst = constraintMatrices_lst
        self.outputVectors_lst = outputVectors_lst
        self.numCurves = numCurves
        self.nDims = nDims
    
    @classmethod
    def from_jointMatrices(cls, ConstraintMatrices: np.ndarray, OutputMatrices: np.ndarray) -> 'separableEqs':
        return separableEqs(_joint2split_constr(ConstraintMatrices), _joint2split_out(OutputMatrices))

    def getConstraintMatrices(self):
        return _split2joint_constr(self.constraintMatrices_lst)
    
    def getOutputMatrices(self):
        return _split2joint_out(self.outputVectors_lst)
        
    def invertProblem(self, solve_for_idx) -> 'separableEqs':
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

        new_constr_lst = []
        new_out_lst = []
        for i in range(self.nDims):
            new_constr, new_out = invertCase(self.constraintMatrices_lst[i], self.outputVectors_lst[i])
            new_constr_lst.append(new_constr)
            new_out_lst.append(new_out)
        return separableEqs(new_constr_lst, new_out_lst)

def _split2joint_constr(constraintMatrices_lst: List[np.ndarray]) -> np.ndarray:
    """
    Generate joint ConstraintMatrices from split constraints.
    
    Parameters:
    - constraintMatrices_lst (List[np.ndarray]): List of split matrices, where each represents the constraints for a given dimension.
    
    Returns:
    np.ndarray: Joint ConstraintMatrices.
    """
    numCurves = constraintMatrices_lst[0].shape[1]
    nDims = len(constraintMatrices_lst)
    assert all([c.shape[1] == numCurves for c in constraintMatrices_lst]), "All constraintMatrices need to have the same number of columns"
    assert sum([c.shape[0] for c in constraintMatrices_lst]) == numCurves-1, "Required N-1 constraints (N is number of curves)"
    
    ConstraintMatrices = np.zeros((numCurves - 1, numCurves, nDims))
    current_eq = 0
    for current_dim, c in enumerate(constraintMatrices_lst):
        if c.size > 0:
            numConstr = c.shape[0]
            ConstraintMatrices[current_eq:current_eq+numConstr, :, current_dim] = c
            current_eq += numConstr
    return ConstraintMatrices

def _split2joint_out(outputVectors_lst: List[np.ndarray]) -> np.ndarray:
    """
    Generate joint OutputMatrices from split outputs.
    
    Parameters:
    - outputVectors_lst (List[np.ndarray]): List of split vectors, where each represents the output for a given dimension.
    
    Returns:
    np.ndarray: Joint OutputMatrices.
    """
    numCurves = outputVectors_lst[0].size
    nDims = len(outputVectors_lst)
    assert all([c.size == numCurves for c in outputVectors_lst]), "All outputVectors need to have the same length as columns in constraintMatrices"
    OutputMatrices = np.zeros((nDims, numCurves, nDims))
    for i, c in enumerate(outputVectors_lst):
        OutputMatrices[i,:,i] = c
    return OutputMatrices

def _findCases(A: np.ndarray, index: int) -> np.ndarray:
        non_zero_idx = np.where(np.any(A[:, :, index] != 0.0, axis=1))[0]
        for i in non_zero_idx:
            mask = np.ones_like(A[i], dtype=bool) 
            mask[:, index] = False
            assert np.all(A[i][mask] == 0), "Mixed matrix, cannot decompose"

        return non_zero_idx

def _joint2split_constr(ConstraintMatrices: np.ndarray) -> List[np.ndarray]:
    """
    Generate separated constrait matrices from joint ConstraintMatrices.
    
    Parameters:
    - ConstraintMatrices (np.ndarray): Constraint matrices.
    
    Returns:
    List[np.ndarray]: List of split matrices, where each represents the constraints for a given dimension.
    """
    numCurves = ConstraintMatrices.shape[1]
    nDims = ConstraintMatrices.shape[2]
    assert ConstraintMatrices.shape[0] == numCurves-1, "Required N-1 constraints (N is number of curves)"

    constraintMatrices_lst = []
    for i in range(nDims):
        idx = _findCases(ConstraintMatrices, i)
        constraintMatrices_lst.append(ConstraintMatrices[idx, :, i])

    return constraintMatrices_lst

def _joint2split_out(OutputMatrices: np.ndarray) -> List[np.ndarray]:
    """
    Generate separated output vectors from joint OutputMatrices..
    
    Parameters:
    - ConstraintMatrices (np.ndarray): Output matrices.
    
    Returns:
    List[np.ndarray]: List of split vectors, where each represents the output for a given dimension.
    """
    nDims = OutputMatrices.shape[2]
    assert OutputMatrices.shape[0] == nDims, "Required as many outputVectors as dimensions"

    outputVectors_lst = []
    for i in range(nDims):
        idx = _findCases(OutputMatrices, i)
        assert len(idx) == 1, "Only one element in OutputMatrices should represent a each dimension"
        outputVectors_lst.append(OutputMatrices[idx[0], :, i])

    return outputVectors_lst

if __name__ == "__main__":
    N = 3
    d = 2
    constraint_matrices = np.zeros((N-1, N, d))
    output_matrices = np.zeros((d, N, d))

    constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
    constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
    output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
    output_matrices[1,:,1] = np.array([1.0,1.0,0.0])

    eqs = separableEqs.from_jointMatrices(constraint_matrices, output_matrices)
    print("constraintMatrices_lst\n",eqs.constraintMatrices_lst)
    print("outputVectors_lst\n",eqs.outputVectors_lst)