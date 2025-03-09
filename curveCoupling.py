import numpy as np
from scipy import optimize, interpolate
from curveInterpExtrapFunc import ndcurve, ndcurve_matrix
import itertools
from auxFunc import my_null_space
from typing import *

_INITIAL_OPT = Literal["fix_largest", "min_dist", "off"]
_INITIAL_OPT_EQ = Literal["mean_val","fix_largest", "min_dist", "off"]
# Type alias for curve input

class curveCouplingProblem:
    def __init__(self, curves: List[ndcurve],
                 ConstraintMatrices: np.ndarray,
                 OutputMatrices: Optional[np.ndarray] = None,
                 ConstraintConstantVector: Optional[np.ndarray] = None,
                 OutputConstantVector: Optional[np.ndarray] = None):
        self.curves = curves
        self.ConstraintMatrices = ConstraintMatrices
        self.OutputMatrices = OutputMatrices
        self.ConstraintConstantVector = ConstraintConstantVector
        self.OutputConstantVector = OutputConstantVector
        self.numCurves = len(curves)
        assert self.numCurves >= 2, "At least two curves are needed"
        self.curves_all = ndcurve_matrix(curves)

        evals = [f(0) for f in curves]
        self.Ndims = set([e.size if isinstance(e,np.ndarray) else 1 for e in evals])
        assert len(self.Ndims) == 1, "All curves must have the same dimensionality"
        self.Ndims = self.Ndims.pop()

        if self.OutputMatrices is None:
            self.OutputMatrices = np.zeros((self.Ndims, self.numCurves, self.Ndims))
            for i in range(self.Ndims):
                self.OutputMatrices[i,:,i] = np.ones(self.numCurves) / self.numCurves

        assert self.ConstraintMatrices.shape[2] == self.Ndims, "Constraint dim 2 must be the number of dimensions"
        assert self.OutputMatrices.shape[2] == self.Ndims, "Output dim 2 must be the number of dimensions"
        assert self.ConstraintMatrices.shape[1] == self.numCurves, "Constraint dim 1 must hbe the number of curves"
        assert self.OutputMatrices.shape[1] == self.numCurves, "Constraint dim 1 must hbe the number of curves"

        if self.ConstraintConstantVector is None:
            self.ConstraintConstantVector = np.zeros(self.ConstraintMatrices.shape[0])
        if self.OutputConstantVector is None:
            self.OutputConstantVector = np.zeros(self.OutputMatrices.shape[0])

    def computeConstraint_from_values(self, vals: np.ndarray) -> np.ndarray:
        if vals.ndim<2:
            vals = vals.reshape((-1,1))
        return np.einsum('ijk,jk->i', self.ConstraintMatrices, vals)+self.ConstraintConstantVector
    
    def computeConstraint(self, params: np.ndarray) -> np.ndarray:
        return self.computeConstraint_from_values(self.curves_all(params))
    
    def computeConstraintJac(self, params: np.ndarray, nu: int = 1) -> np.ndarray:
        vals = self.curves_all(params, nu=nu)
        if vals.ndim<2:
            vals = vals.reshape((-1,1))
        return np.einsum('ijk,jk->ij', self.ConstraintMatrices, vals)
    
    def computeTangent(self, params: np.ndarray):
        J = self.computeConstraintJac(params)
        nullspace = my_null_space(J)
        return nullspace[:,0]
    
    def computeOutput_from_values(self, vals: np.ndarray) -> np.ndarray:
        if vals.ndim<2:
            vals = vals.reshape((-1,1))
        return np.einsum('ijk,jk->i', self.OutputMatrices, vals)+self.OutputConstantVector
    
    def computeOutput(self, params: np.ndarray) -> np.ndarray:
        return self.computeOutput_from_values(self.curves_all(params))
    
class curveCouplingProblem_Equality:
    def __init__(self, curves: List[ndcurve], match_index: Optional[int] = None):
        self.curves = curves
        self.curves_match_index = [c.extractIndex(match_index) for c in curves]
        self.match_index = match_index
        self.numCurves = len(curves)
        assert self.numCurves >= 2, "At least two curves are needed"
        evals = [f(0) for f in curves]
        self.Ndims = set([e.size if isinstance(e,np.ndarray) else 1 for e in evals])
        assert len(self.Ndims) == 1, "All curves must have the same dimensionality"
        self.Ndims = self.Ndims.pop()
        self.curves_all = ndcurve_matrix(curves)
        self.curves_all_match_index = self.curves_all.extractIndex(match_index)

    def computeConstraint_from_values(self, vals: np.ndarray, fixed_index: Optional[int] = None) -> np.ndarray:
        if fixed_index is None:
            return vals[:-1]-vals[1:]
        else:
            val_fixed = vals[fixed_index]
            vals_others = np.delete(vals, fixed_index)
            return vals_others - val_fixed
        
    def computeConstraint(self, params: np.ndarray, fixed_index: Optional[int] = None) -> np.ndarray:
        return self.computeConstraint_from_values(self.curves_all_match_index(params), fixed_index)
    
    def computeConstraintJac(self, params, fixed_index: Optional[int] = None, nu=1):
        df = self.curves_all_match_index(params, nu=nu)
        J_val = np.zeros([self.numCurves-1, self.numCurves])
        if fixed_index is None:
            J_val[np.arange(self.numCurves-1), np.arange(self.numCurves-1)] = df[np.arange(self.numCurves-1)]
            J_val[np.arange(self.numCurves-1), np.arange(1,self.numCurves)] = -df[np.arange(1,self.numCurves)]
        else:
            other_indices = np.delete(np.arange(self.numCurves), fixed_index)
            J_val[np.arange(self.numCurves-1), other_indices] = df[other_indices]
            J_val[:, fixed_index] = -df[fixed_index]
        return J_val
    
    def computeTangent(self, params: np.ndarray):
        J = self.computeConstraintJac(params)
        nullspace = my_null_space(J)
        return nullspace[:,0]
    
    def computeOutput_from_values(self, vals: np.ndarray) -> np.ndarray:
        return np.mean(vals, axis=0)
    
    def computeOutput(self, params: np.ndarray) -> np.ndarray:
        return self.computeOutput_from_values(self.curves_all(params))
    
    def to_General(self, fixed_index: Optional[int] = None):
        ConstraintMatrices = np.zeros((self.numCurves-1, self.numCurves, self.Ndims))
        match_index = 0 if self.match_index is None else self.match_index
        if fixed_index is None:
            ConstraintMatrices[np.arange(self.numCurves-1), np.arange(self.numCurves-1), match_index] = 1.0
            ConstraintMatrices[np.arange(self.numCurves-1), np.arange(1,self.numCurves), match_index] = -1.0
        else:
            other_indices = np.delete(np.arange(self.numCurves), fixed_index)
            ConstraintMatrices[np.arange(self.numCurves-1), other_indices, match_index] = 1.0
            ConstraintMatrices[:, fixed_index, match_index] = -1.0
        return curveCouplingProblem(self.curves, ConstraintMatrices)




def solveCurveCoupling_bruteForce(
    prb: curveCouplingProblem,
    tolerance: float = 0.01,
    iter_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the semiaddition of parametric curves by brute force.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        tolerance (float): Tolerance in solving the equality condition.
        iter_points (int): Number of iteration points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output curve and results in parametric space.
    """

    array_params = np.linspace(0.0, 1.0, iter_points)
    array_curves = [c(array_params) for c in prb.curves]
    combinations = itertools.product(range(iter_points), repeat=prb.numCurves)
    Output = []
    Results = []
    for comb in combinations:
        curve_vals = np.array([a[i] for a,i in zip(array_curves,reversed(comb))])
        match_vals = curve_vals[:,prb.curves_all.match_index]
        if np.linalg.norm(prb.computeConstraint_from_values(match_vals)) < tolerance:
            param = np.array([array_params[i] for i in reversed(comb)])
            out_val = prb.computeOutput_from_values(curve_vals)
            Results.append(param)
            Output.append(out_val)
    
    return np.array(Output), np.array(Results)

def solveCurveCoupling_bruteForce_localSolve(
    prb: curveCouplingProblem,
    iter_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the semiaddition of parametric curves by brute force with local solving.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output curve and results in parametric space.
    """

    array_params = np.linspace(0.0, 1.0, iter_points)
    combinations = itertools.product(range(iter_points), repeat=prb.numCurves)
    Output = []
    Results = []

    def f_opt(x):
        y = np.concatenate([prb.computeConstraint(x), [0.0]])
        J = np.vstack([prb.computeConstraintJac(x), [np.zeros_like(x)]])
        return y,J
    for comb in combinations:
        param0 = np.array([array_params[i] for i in reversed(comb)])
        res = optimize.root(f_opt, param0, method='hybr', tol=1e-6, jac=True, options={
                            'eps': 0.1, 'diag': [1e-3] * prb.numCurves})
        if res.success and np.all(res.x>=0.0) and np.all(res.x<=1.0):
            Results.append(res.x)
            Output.append(prb.computeOutput(res.x))
    
    return np.array(Output), np.array(Results)


def solveCurveCoupling_bruteForce_matrix(
    prb: curveCouplingProblem,
    iter_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the constraint error magnitude for each parameter.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Difference values and tested parametric space array.
    """

    array_params = np.linspace(0.0, 1.0, iter_points)
    array_curves = [c(array_params) for c in prb.curves]
    array_results = np.zeros([array_params.size] * prb.numCurves)

    for index in np.ndindex(array_results.shape):
        match_vals = np.array([a[i,prb.curves_all.match_index] for a,i in zip(array_curves,reversed(index))])
        array_results[index] = np.linalg.norm(prb.computeConstraint_from_values(match_vals))

    return array_results, array_params

def solveCurveCoupling_Equality(
    prb: curveCouplingProblem_Equality, *,
    solve_init: _INITIAL_OPT_EQ = "min_dist",
    tolerance: float = 1e-9,
    param_start: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the curve coupling problem with equality constraints.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        solve_init (_INITIAL_OPT_EQ): Initial solver option.
        tolerance (float): Tolerance for solving the equality condition.
        param_start (Optional[np.ndarray]): Initial parameters.
        **kwargs: Additional arguments for solveCurveCoupling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output curve and results in parametric space.
    """
    def solveInitVal(param_in: np.ndarray, val: float) -> np.ndarray:
        f_opt = [lambda x, f=f: f(x) - val for f in prb.curves_match_index]
        opt_res = [optimize.root_scalar(
            f, x0=p, x1=p + 1e-3, method='secant') for f, p in zip(f_opt, param_in)]
        return np.array([a.root for a in opt_res])

    def solveInitMinDist(param_in: np.ndarray) -> np.ndarray:
        def f_opt(x): return np.sum((x - param_in) ** 2)
        constraints = [{'type': 'eq', 'fun': lambda x:
                        prb.computeConstraint(x)}]
        opt_res = optimize.minimize(
            f_opt, param_in, constraints=constraints, tol=1e-6)

        return opt_res.x

    def solveInitFixed(param_in: np.ndarray, fixed_index: int = 0) -> np.ndarray:
        param_fixed = param_in[fixed_index]
        index_others = np.delete(np.arange(prb.numCurves), fixed_index)

        def f_opt(x_trunc: np.ndarray):
            x = np.insert(x_trunc, fixed_index, param_fixed)
            y = prb.computeConstraint(x)
            J = prb.computeConstraintJac(x)
            J_trunc = np.delete(J, fixed_index, 1)
            return y, J_trunc
        res = optimize.root(f_opt, param_in[index_others], method='hybr', tol=tolerance, jac=True, options={
                            'eps': 0.1, 'diag': [1e-3] * (prb.numCurves-1)})
        return np.insert(res.x, fixed_index, param_fixed)

    if param_start is None:
        param_start = np.zeros(prb.numCurves, dtype=float)

    if solve_init == "min_dist":
        param_0 = solveInitMinDist(param_start)
    elif solve_init == "fix_largest":
        fixed_curve = np.argmax(prb.curves_all_match_index(param_start))
        param_0 = solveInitFixed(param_start, fixed_curve)
    elif solve_init == "mean_val":
        val = np.mean(prb.curves_all_match_index(param_start))
        param_0 = solveInitVal(param_start, val)
    elif solve_init == "off":
        param_0 = param_start
    else:
        raise Exception("Unknown initial solver")

    return solveCurveCoupling(prb, param_start=param_start, tolerance=tolerance, solve_init="off", **kwargs)


def solveCurveCoupling(
    prb: curveCouplingProblem,
    param_start: Optional[np.ndarray] = None,
    param_final: Optional[np.ndarray] = None,
    param_stop: Optional[np.ndarray] = None,
    stop_circulation: bool = False,
    initial_dir: Optional[np.ndarray] = None,
    solve_init: _INITIAL_OPT = "min_dist",
    step_0: float = 0.0025,
    step_min: float = 0.001,
    step_max: float = 0.005,
    guess_factor: float = 2.0,
    tolerance: float = 1e-9,
    it_max: int = 1e5,
    output_points: Optional[int] = None,
    callbacFunc: Optional[Callable[[np.ndarray, np.ndarray], None]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the semiaddition of parametric curves.
    The result is computed in the parametric space as the points (x1, x2, ...) such that c0(x0)[0] = c1(x1)[0] = ...
    Then, the resulting curve is computed as the points c_res = mean(c0(x0), c1(x1), ...).
    The calculation is made by keeping track of the direction of the previous step in the parametric space
    and trying to find the next solution in the same direction.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        param_start (Optional[np.ndarray]): Initial parameters.
        param_final (Optional[np.ndarray]): Final parameters.
        param_stop (Optional[np.ndarray]): Stop if get close to one of these parameters.
        stop_circulation (bool): Whether to stop if get close to initial point.
        initial_dir (Optional[np.ndarray]): Initial direction.
        solve_init (_INITIAL_OPT): Which solver to use initially.
        step_0 (float): Initial step size.
        step_min (float): Minimum step size.
        step_max (float): Maximum step size.
        guess_factor (float): Scale factor for guess in the desired direction.
        tolerance (float): Tolerance for solving the equality condition.
        it_max (int): Maximum iterations.
        output_points (Optional[int]): Number of output points.
        callbacFunc (Optional[Callable[[np.ndarray, np.ndarray], None]]): Callback function.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Output curve and results in parametric space.
    """
    if param_start is None:
        param_start = np.zeros(prb.numCurves, dtype=float)
    if param_final is None:
        param_final = np.ones(prb.numCurves, dtype=float)

    assert param_start.size == prb.numCurves, "Initial params must have the same number of values as curves"
    assert param_final.size == prb.numCurves, "Final params must have the same number of values as curves"
    assert step_min <= step_max, "step_min should be less than or equal to step_max"

    if param_stop is not None:
        if param_stop.size == 0:
            param_stop = None
        else:
            if param_stop.ndim == 1:
                param_stop = param_stop.reshape((1, -1))
            assert param_stop.shape[1] == prb.numCurves, "Stop params must have the same number of values as curves"

    if initial_dir is not None:
        assert initial_dir.size == prb.numCurves, "Initial_dir must have the same number of values as curves"

    def updateStep(direction: np.ndarray, direction_0: np.ndarray, step: float) -> float:
        corr = np.dot(direction, direction_0)
        alpha = 1.5 / (5 * ((1 - corr) ** 2) + 1)
        return max(min(step * alpha, step_max), step_min)

    def solveInitFixed(param_in: np.ndarray, fixed_index: int = 0) -> np.ndarray:
        param_fixed = param_in[fixed_index]
        index_others = np.delete(np.arange(prb.numCurves), fixed_index)

        def f_opt(x_trunc: np.ndarray):
            x = np.insert(x_trunc, fixed_index, param_fixed)
            y = prb.computeConstraint(x)
            J = prb.computeConstraintJac(x)
            J_trunc = np.delete(J, fixed_index, 1)
            return y, J_trunc
        res = optimize.root(f_opt, param_in[index_others], method='hybr', tol=tolerance, jac=True, options={
                            'eps': 0.1, 'diag': [1e-3] * (prb.numCurves-1)})
        return np.insert(res.x, fixed_index, param_fixed)

    def solveInitMinDist(param_in: np.ndarray) -> np.ndarray:
        def f_opt(x): return np.sum((x - param_in) ** 2)
        constraints = [{'type': 'eq', 'fun': lambda x:
                        prb.computeConstraint(x)}]
        opt_res = optimize.minimize(
            f_opt, param_in, constraints=constraints, tol=1e-6)

        return opt_res.x

    def solveMult(param_guess: np.ndarray, param_prev: np.ndarray, dist_tgt: float) -> np.ndarray:
        def evalParam(x: np.ndarray, compute_jac: bool = True) -> np.ndarray:
            dist = np.linalg.norm(x - param_prev)

            if compute_jac:
                J_dist = (x - param_prev) / dist
                J_tot = np.vstack([prb.computeConstraintJac(x), J_dist])
                return np.concatenate((prb.computeConstraint(x), [dist - dist_tgt])), J_tot

            return np.concatenate((prb.computeConstraint(x), [dist - dist_tgt]))

        res = optimize.root(evalParam, param_guess, method='hybr', tol=tolerance, jac=True, options={
                            'eps': 0.1, 'diag': [dist_tgt] * prb.numCurves})
        return res.x

    def reparametrizeCurve(c: np.ndarray, nsteps: int) -> np.ndarray:
        x_interp = np.linspace(0.0, 1.0, num=c.shape[0])
        f_interp = interpolate.CubicSpline(
            x_interp, c, axis=0, bc_type="natural")
        return np.array([f_interp(x) for x in np.linspace(0.0, 1.0, num=nsteps)])

    def computeTangent(params: np.ndarray, prev_dir: np.ndarray) -> np.ndarray:
        tangent = prb.computeTangent(params)
        if np.dot(prev_dir, tangent) < 0:
            tangent *= -1.0
        return tangent

    if solve_init == "min_dist":
        param_prev = solveInitMinDist(param_start)
    elif solve_init == "fix_largest":
        fixed_curve = np.argmax(param_start)
        param_prev = solveInitFixed(param_start, fixed_curve)
    elif solve_init == "off":
        param_prev = param_start
    else:
        raise Exception("Unknown initial solver")

    if stop_circulation:
        if param_stop is None:
            param_stop = param_prev.reshape((1, -1))
        else:
            param_stop = np.vstack([param_stop, param_prev])

    Res = [param_prev]
    Output = [prb.computeOutput(param_prev)]

    if initial_dir is None:
        direction_0 = computeTangent(param_prev, np.ones(prb.numCurves))
    else:
        direction_0 = initial_dir

    step = step_0
    param_guess = param_prev + direction_0 * step * guess_factor

    dist_stop_max = 0.0
    flag_stop = False
    it = 0

    while np.all(param_prev < param_final) and it < it_max:
        try:
            it += 1
            param = solveMult(param_guess, param_prev, step)

            if param_stop is not None:
                aux_vals = [(np.linalg.norm(param - p_stop), p_stop)
                            for p_stop in param_stop]
                dist_stop, close_param_stop = min(aux_vals, key=lambda x: x[0])
                if dist_stop > dist_stop_max:
                    dist_stop_max = dist_stop
                else:
                    if dist_stop < step:
                        param = close_param_stop
                        flag_stop = True

            Res.append(param)
            Output.append(prb.computeOutput(param))
            direction = computeTangent(param, direction_0)

            step = updateStep(direction, direction_0, step)
            param_guess = param_prev + direction * step * guess_factor

            param_prev = param
            direction_0 = direction

            if callbacFunc is not None:
                callbacFunc(np.array(Output), np.array(Res))
            if flag_stop:
                break
        except Exception as e:
            print(f"Exception: {e}")
            break

    if output_points is None:
        return np.array(Output), np.array(Res)
    if output_points <= 0:
        c_sizes = [c.shape[0] for c in prb.curves]
        output_points = sum(c_sizes)

    Output = reparametrizeCurve(np.array(Output), output_points)
    Res = reparametrizeCurve(np.array(Res), output_points)

    return Output, Res

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    from curveGenerators import *

    p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
    p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
    p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                   [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_peaks(pts, 200) for pts in points]
    match_index = 1

    curves = ndcurve.createList(data)
    prob_eq = curveCouplingProblem_Equality(curves, match_index)

    params = np.array([0.1, 0.2, 0.3])
    fixed_index = None
    print(prob_eq.computeConstraint(params, fixed_index))
    print(prob_eq.computeConstraintJac(params, fixed_index))
    print(prob_eq.computeOutput(params))

    prob = prob_eq.to_General(fixed_index)
    print(prob.computeConstraint(params))
    print(prob.computeConstraintJac(params))
    print(prob.computeOutput(params))

    out, res = solveCurveCoupling_Equality(prob_eq)
    out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=20)

    fig = plt.figure()
    plot_h = 2
    gs = gridspec.GridSpec(2, plot_h * len(data))
    axs = []

    for i in range(0, len(data)):
        axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, len(data):]))
    axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])

    axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
    axs[-1].scatter(res_brute[:, 0], res_brute[:, 1], res_brute[:, 2], color='r', marker ='.',alpha=0.1)

    axs[-2].plot(out[:, 0], out[:, 1])
    axs[-2].scatter(out_brute[:, 0], out_brute[:, 1], color='r', marker ='.',alpha=0.1)

    plt.pause(0.1)
    input("Press Enter")





    p0 = np.array([[0.0, 0.0], [0.55,0.6], [1.1, 0.88], [1.27, 0.72], [1.1,0.55]])
    p0 = np.concatenate([p0, [2.0,1.0]-np.flip(p0,axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0,1.0]-np.flip(p1,axis=0)])
    p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_CubicSpline(pts, 200) for pts in points]

    curves = ndcurve.createList(data)
    constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
    output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

    constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
    constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
    output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
    output_matrices[1,:,1] = np.array([1.0,1.0,0.0])
    
    prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)

    out, res = solveCurveCoupling(prob)
    out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)

    fig = plt.figure()
    plot_h = 2
    gs = gridspec.GridSpec(2, plot_h * len(data))
    axs = []

    for i in range(0, len(data)):
        axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, len(data):]))
    axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])

    axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
    axs[-1].scatter(res_brute[:, 0], res_brute[:, 1], res_brute[:, 2], color='r', marker ='.',alpha=0.1)

    axs[-2].plot(out[:, 0], out[:, 1])
    axs[-2].scatter(out_brute[:, 0], out_brute[:, 1], color='r', marker ='.',alpha=0.1)

    plt.pause(0.1)
    input("Press Enter")

