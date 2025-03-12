from matplotlib import pyplot as plt

folder = "assets\\"
extension = ".png"

def make_fig(names):
    run()
    for i, name in zip(plt.get_fignums(), names):  # Get all figure numbers
        plt.figure(i)
        plt.savefig(folder+name+extension)
        print("Figure saved as "+name+extension)
    plt.close('all')


from Test_curveGenerator import run
make_fig(["curveGenerator"])

from Test_curveInterpolator import run
make_fig(["curveInterpolator"])

# from Test_curveCoupling_Equality import run
# make_fig(["curveCoupling_Equality"])

# from Test_curveCoupling import run
# make_fig(["curveCoupling_General"])

# from Test_curveCoupling_Islands_Equality import run
# make_fig(["curveCoupling_Islands_Equality"])

# from Test_curveCoupling_Islands import run
# make_fig(["curveCoupling_Islands_General"])

# from Test_curveCoupling_Singularity_Equality import run
# make_fig(["curveCoupling_Singularities_Equality"])

# from Test_curveCoupling_Singularity import run
# make_fig(["curveCoupling_Singularities_General"])

# from Test_stabilityAnalysis_single import run
# make_fig(["curveCoupling_Stability_Single"])

# from Test_stabilityAnalysis_Equality import run
# make_fig(["curveCoupling_Stability_Equality"])

# from Test_stabilityAnalysis import run
# make_fig(["curveCoupling_Stability_General"])

from Test_inverseProblem import run
make_fig(["curveCoupling_Direct_Problem","curveCoupling_Inverse_Problem_0","curveCoupling_Inverse_Problem_1","curveCoupling_Inverse_Problem_2"])

plt.show(block=False)
print("Done!")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.