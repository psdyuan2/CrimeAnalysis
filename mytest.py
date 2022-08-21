import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq
import os
from causallearn.utils.GraphUtils import GraphUtils
def test_bnlearn_discrete_datasets():
    benchmark_names = [
        "asia"
    ]

    bnlearn_path = '/Users/dawson/Desktop/CausalLearning/Causal_learning/causal-learn-main/tests/TestData/bnlearn_discrete_10000'
    for bname in benchmark_names:
        data = np.loadtxt(os.path.join(bnlearn_path, f'{bname}.txt'), skiprows=1)
        G, edges = fci(data, chisq, 0.05, verbose=False)
        print('finish')
    return G, edges
if __name__ == "__main__":
    G = test_bnlearn_discrete_datasets()
    pdy = GraphUtils.to_pydot(G)
    pdy.write_png('test.png')
