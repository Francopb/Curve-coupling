from Test_inverseProblem import run as run_inv
from Test_stabilityAnalysis import run as run_stab
from Test_stabilityAnalysis_Equality import run as run_stab_eq
from Test_stabilityAnalysis_single import run as run_stab_single
from Test_curveCoupling_Singularity import run as run_sing
from Test_curveCoupling_Singularity_Equality import run as run_sing_eq
from Test_curveCoupling_Islands import run as run_island
from Test_curveCoupling_Islands_Equality import run as run_island_eq
from Test_curveCoupling import run as run_coupling
from Test_curveCoupling_Equality import run as run_coupling_eq
from Test_curveInterpolator import run as run_interp
from Test_curveGenerator import run as run_gen
from Test_curveCoupling_FindIslands import run as run_findIslands
from matplotlib import pyplot as plt

folder = "assets\\"
extension = ".png"


def make_fig(run, names):
    run()
    for i, name in zip(plt.get_fignums(), names):  # Get all figure numbers
        plt.figure(i)
        plt.savefig(folder+name+extension)
        print("Figure saved as "+name+extension)
    plt.close('all')


# make_fig(run_gen, ["curveGenerator"])
# make_fig(run_interp, ["curveInterpolator"])
# make_fig(run_coupling_eq, ["curveCoupling_Equality"])
# make_fig(run_coupling, ["curveCoupling_General"])
# make_fig(run_island_eq, ["curveCoupling_Islands_Equality"])
# make_fig(run_island, ["curveCoupling_Islands_General"])
# make_fig(run_sing_eq, ["curveCoupling_Singularities_Equality"])
# make_fig(run_sing, ["curveCoupling_Singularities_General"])
# make_fig(run_stab_single, ["curveCoupling_Stability_Single"])
# make_fig(run_stab_eq, ["curveCoupling_Stability_Equality"])
# make_fig(run_stab, ["curveCoupling_Stability_General"])
# make_fig(run_inv, ["curveCoupling_Direct_Problem", "curveCoupling_Inverse_Problem_0",
#          "curveCoupling_Inverse_Problem_1", "curveCoupling_Inverse_Problem_2"])
make_fig(run_findIslands, ["curveCoupling_findIslands_General",
                           "curveCoupling_findIslands_General_zoom"])

plt.show(block=False)
print("Done!")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
