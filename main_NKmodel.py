import argparse
import numpy as np
import torch
import random
import itertools

from main import COMBO
from COMBO.experiments.test_functions.experiment_configuration import sample_init_points


NKMODEL_N_STAGES = 6
INITIAL_POINTS_N = 2  # if it is <= 1, it raises ZeroDivisionError. (왜?)
NKMODEL_DATAPATH = "/Users/hanseul_jo/Desktop/Post AI/Exp input/Game1 landscape.txt"

def _generate_random_seeds(seed_str, n_test_case_seed=5, n_init_point_seed=5, seed_num=3):
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_test_case_seed):
        result[tuple(rng_state.randint(0, 10000, (seed_num-1,)))] = list(rng_state.randint(0, 10000, (n_init_point_seed,)))
    return result

def generate_random_seed_pair_nkmodel():
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    return _generate_random_seeds(seed_str="NKMODEL", n_test_case_seed=5, n_init_point_seed=5, seed_num=3)


class NKmodel(object):
    """
    NKmodel class. A single NK model.
    Huge thanks to https://github.com/elplatt/nkmodel.git
    
    <PARAM>
    N: the number of loci
    K: the number of the other dependent loci for each locus (0 <= K <= N-1)
    A: the number of states that each locus can have (e.g.: 2 for binary variables)
    """
    def __init__(self, N, K, A, interdependence=None, contributions=None, random_seeds=(None, None)):
        assert 0 <= K <= N-1
        self.N = N
        self.K = K
        self.A = A
        self.loci = range(N)
        if interdependence is None:
            # randomly generated interdependence matrix
            self.interdependence = np.full((N,N), False)  #numpy 1.8
            rng_state_dep = np.random.RandomState(seed=random_seeds[0])
            for i in self.loci:
                dependence = [i] + list(rng_state_dep.choice(list(set(self.loci) - set([i])), size=K, replace=False))
                self.interdependence[i][dependence] = True
        else:
            self.interdependence = interdependence
        if contributions is None:
            self.contributions = [{} for _ in self.loci]
            rng_state_ctrb = np.random.RandomState(seed=random_seeds[1])
            for i in range(N):
                for label in itertools.product(range(A), repeat=K+1):  # K+1 subcollection of loci values that effects the locus i
                    self.contributions[i][label] = float(rng_state_ctrb.random())  # between [0,1]   --> 다르게도 할 수 있지 않을까
        else:
            self.contributions = contributions

    def fitness_and_contributions(self, state, negative=False):
        """
        Given a state(: a tuple/string of length N), 
        Return fitness value and a list of contributions of each loci.
        """
        ctrbs = []
        if type(state) == str:
            state = np.array([int(state[i]) for i in range(self.N)])
        else:
            state = np.array(state)
        for i in self.loci:
            label = tuple(state[self.interdependence[i]])
            ctrbs.append(self.contributions[i][label])
        fitness_value = sum(ctrbs)
        if negative:
            fitness_value = -fitness_value
        return fitness_value, ctrbs
    
    def fitness(self, state, negative=False):
        f, _ = self.fitness_and_contributions(state, negative=negative)
        return f
    
    def landscape(self, negative = False):
        """
        Return a dictionary mapping each state to its fitness value. (Naive algorithm)
        """
        result_dic = {}
        for state in itertools.product(range(self.A), repeat=self.N): # along all possible states
            fitness, _= self.fitness_and_contributions(state)
            if negative:
                fitness = -fitness
            result_dic[state] = fitness
        return result_dic

    def landscape_with_contributions(self):
        """
        Return a dictionary mapping each state to its fitness value and contributions of loci. (Naive algorithm)
        """
        return {state: self.fitness_and_contributions(state) for state in itertools.product(range(self.A), repeat=self.N)}  # along all possible states

    def local_search(self, state):
        """
        Given a state(: a tuple/string of length N),
        Generate "changes" of fitness values when each of loci modified from the state.
        Return a list of length N.
        """
        raise NotImplementedError

    def get_global_optimum(self, negative=False):
        """
        Global maximum fitness value and its maximizer(state), in a NAIVE way.
        """
        landscape = self.landscape()
        optimum = max(landscape.values())
        states = [s for s in landscape.keys() if landscape[s] == optimum]
        if negative:
            optimum = -optimum
        return optimum, states
        
    def get_optimum_and_more(self, order, negative=False):
        """
        First several maximum fitness values and their maximizers(states), in a NAIVE way.
        """
        landscape_list = sorted(self.landscape().items(), key=lambda x: -x[1])
        optima2states = {}
        cnt = 0
        for state, fitness in landscape_list:
            if negative:
                fitness = -fitness
            if fitness in optima2states:
                optima2states[fitness].append(state)
            else:
                cnt += 1
                if cnt > order:
                    break
                optima2states[fitness] = [state]
        return optima2states
    
    def print_info(self, path=None):
        if path is None:
            raise NotADirectoryError
        with open(path + "knowledge.txt", "w") as f1:
            for i in range(self.N):
                f1.write("".join(["O" if b else "X" for b in self.interdependence[i]]) + "\n")
        with open(path + "landscape.txt", "w") as f2:
            d = self.landscape_with_contributions()
            for state, (fit, ctrbs) in d.items():
                ctrbs = [str(round(v, 4)) for v in ctrbs]
                fit = str(round(fit, 4))
                state = "".join([str(x) for x in state])
                f2.write("\t".join([state] + ctrbs + [fit]) + "\n")
            optlist = list(self.get_optimum_and_more(order=10).items())
            optlist.sort(key=lambda x: -x[0])
            for i in range(10):
                opt, optstates = optlist[i]
                f2.write(f"{i+1}-th optimum: {opt} {optstates}\n")


class NK_COMBO(object):
    """
    Preprocessing NK model to solve it with COMBO
    """
    def __init__(self, N, K, A=2, im=None, ctrbs=None, random_seeds=(None, None, None)):
        self.n_vertices = np.repeat(A, N)
        self.suggested_init = sample_init_points(self.n_vertices, INITIAL_POINTS_N, random_seed=random_seeds[-1])
        #self.suggested_init = torch.empty(0).long()
        #self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, INITIAL_POINTS_N - self.suggested_init.size(0), random_seed=random_seeds[-1])], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seeds[i]).zfill(4) if random_seeds[i] is not None else 'None' for i in range(len(random_seeds))])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        self.nkmodel = NKmodel(N, K, A, interdependence=im, contributions=ctrbs, random_seeds=random_seeds[:2])
        self.nkmodel.print_info(path="/Users/hanseul_jo/Desktop/Post AI/COMBO/")

    def evaluate(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)

    def _evaluate_single(self, x):
        #assert x.dim() == 1
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = self.nkmodel.fitness(tuple(x), negative=True)
        return torch.Tensor([evaluation])  # 1 by 1 Tensor


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='COMBO : Combinatorial Bayesian Optimization using the graph Cartesian product')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=1)
    parser_.add_argument('--dir_name', dest='dir_name')
    parser_.add_argument('--objective', dest='objective', default='nkmodel')
    parser_.add_argument('--random_seed_config', dest='random_seed_config', type=int, default=None)
    parser_.add_argument('--parallel', dest='parallel', action='store_true', default=False)
    parser_.add_argument('--device', dest='device', type=int, default=None)
    parser_.add_argument('--task', dest='task', type=str, default='both')

    args_ = parser_.parse_args()
    print(args_)
    kwag_ = vars(args_)
    dir_name_ = kwag_['dir_name']
    objective_ = kwag_['objective']
    random_seed_config_ = kwag_['random_seed_config']
    parallel_ = kwag_['parallel']
    if args_.device is None:
        del kwag_['device']
    print(kwag_)
    if random_seed_config_ is not None:
        assert 1 <= int(random_seed_config_) <= 25
        random_seed_config_ -= 1
    assert (dir_name_ is None) != (objective_ is None)

    if objective_ == 'nkmodel':
        random_seed_pair_ = generate_random_seed_pair_nkmodel()
        case_seed_ = sorted(random_seed_pair_.keys())[int(random_seed_config_ / 5)]
        init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
        kwag_['objective'] = NK_COMBO(6, 1, random_seeds=(case_seed_[0], case_seed_[1], init_seed_))
    else:
        if dir_name_ is None:
            raise NotImplementedError
    COMBO(**kwag_)
