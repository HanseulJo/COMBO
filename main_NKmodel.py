import argparse
import numpy as np
import torch
import os
import sys
import itertools
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from time import time

from main import COMBO
from COMBO.experiments.test_functions.experiment_configuration import sample_init_points

INITIAL_POINTS_N = 2  # if it is <= 1, it raises ZeroDivisionError.

# If USE_DATA = True, the NK_COMBO will use the data
# Change the paths if you want.
USE_DATA = False
GAME_NUM = 4
NKMODEL_DATAPATH = f"../Exp input/Game{GAME_NUM} landscape.txt"
NKMODEL_IM_PATH = f"../Exp input/Game{GAME_NUM} knowledge.txt"

# If RECORD = True, the result will be recorded in a file
RECORD = True

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

def text_to_interdependence(path):
    with open(path, "r") as f:
        im = np.stack(tuple([[l == "X" for l in line.strip()] for line in f]))
    return im

def text_to_landscape(path):
    landscape = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            k = tuple([int(x) for x in line[0]])
            fit = int(line[-1])
            ctrbs = [int(x) for x in line[1:-1]]
            landscape[k] = (fit, ctrbs)
    return landscape

def im_landscape_to_contributions(im, landscape, A=2):
    N = im.shape[0]
    K = im[0].sum() - 1
    ctrbs = [{} for _ in range(N)]
    for i in range(N):
        for state in itertools.product(*[range(A) if x else [0] for x in list(im[i])]):  
            label = tuple([state[j] for j in range(N) if im[i][j]])
            ctrbs[i][label] = landscape[state][1][i]
    return ctrbs

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
            self.interdependence = np.full((N,N), False)
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
                    self.contributions[i][label] = float(rng_state_ctrb.randint(0,30))  # integers [0, 30]: Can be modified.
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

    def get_global_optimum(self, negative=False, cache=False, given_landscape=None):
        """
        Global maximum fitness value and its maximizer(state), in a NAIVE way.
        """
        landscape = self.landscape() if given_landscape is None else given_landscape
        optimum = max(landscape.values())
        states = [s for s in landscape.keys() if landscape[s] == optimum]
        if negative:
            optimum = -optimum
        if cache:
            return optimum, states, landscape
        else:
            return optimum, states
        
    def get_optimum_and_more(self, order, negative=False, cache=False, given_landscape=None):
        """
        First several maximum fitness values and their maximizers(states), in a NAIVE way.
        """
        landscape = self.landscape() if given_landscape is None else given_landscape
        landscape_list = sorted(landscape.items(), key=lambda x: -x[1])
        state_opt, fit_opt = landscape_list[0]
        if negative:
            fit_opt = -fit_opt
        optima2states = [{"fitness": fit_opt, "states":[state_opt]}]
        cnt = 1
        for state, fitness in landscape_list[1:]:
            if negative:
                fitness = -fitness
            if fitness == optima2states[-1]["fitness"]:
                optima2states[-1]["states"].append(state)
            else:
                cnt += 1
                if cnt > order:
                    break
                optima2states.append({"fitness": fitness, "states":[state]})
        if cache:
            return optima2states, landscape
        else:
            return optima2states
    
    def print_info(self, path=None):
        if path is None:
            print("\nInterdependence Matrix:")
            for i in range(self.N):
                print("".join(["X" if b else "O" for b in self.interdependence[i]]))
            print("\nLandscape:")
            d = self.landscape_with_contributions()
            for state, (fit, ctrbs) in d.items():
                ctrbs = [str(round(v, 4)) for v in ctrbs]
                fit = str(round(fit, 4))
                state = "".join([str(x) for x in state])
                print("\t".join([state] + ctrbs + [fit]))
            print("===")
            optlist = self.get_optimum_and_more(order=10)
            for i in range(10):
                opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                print(f"{i+1}-th optimum: {opt} {optstates}")
        else:
            with open(path + "/knowledge.txt", "w") if path is not None else sys.stdout as f1:
                for i in range(self.N):
                    print("".join(["X" if b else "O" for b in self.interdependence[i]]), file=f1)
            with open(path + "/landscape.txt", "w") if path is not None else sys.stdout as f2:
                d = self.landscape_with_contributions()
                for state, (fit, ctrbs) in d.items():
                    ctrbs = [str(round(v, 4)) for v in ctrbs]
                    fit = str(round(fit, 4))
                    state = "".join([str(x) for x in state])
                    print("\t".join([state] + ctrbs + [fit]), file=f2)
                print("===", file=f2)
                optlist = self.get_optimum_and_more(order=10)
                for i in range(10):
                    opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                    print(f"{i+1}-th optimum: {opt} {optstates}", file=f2)
    
    def rank_by_fitness(self, fitness_value, given_landscape=None):
        raise NotImplementedError
        


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
        evaluation = self.nkmodel.fitness(tuple(x), negative=True)  # To solve minimization problem, "negative=True."
        return torch.Tensor([evaluation])  # 1 by 1 Tensor


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='COMBO : Combinatorial Bayesian Optimization using the graph Cartesian product')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=1)
    parser_.add_argument('--N', dest='N', type=int, default=6)
    parser_.add_argument('--K', dest='K', type=int, default=1)
    parser_.add_argument('--A', dest='A', type=int, default=2)
    parser_.add_argument('--dir_name', dest='dir_name')
    parser_.add_argument('--objective', dest='objective', default='nkmodel')
    parser_.add_argument('--random_seed_config', dest='random_seed_config', type=int, default=None)
    parser_.add_argument('--parallel', dest='parallel', action='store_true', default=False)
    parser_.add_argument('--device', dest='device', type=int, default=None)
    parser_.add_argument('--task', dest='task', type=str, default='both')
    parser_.add_argument('--game_num', dest='game_num', type=int, default=-1)

    args_ = parser_.parse_args()

    if args_.game_num != -1:
        USE_DATA = True
        GAME_NUM = args_.game_num
        NKMODEL_DATAPATH = f"../Exp input/Game{GAME_NUM} landscape.txt"
        NKMODEL_IM_PATH = f"../Exp input/Game{GAME_NUM} knowledge.txt"

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
        im, ctrbs = None, None
        if USE_DATA:
            im = text_to_interdependence(NKMODEL_IM_PATH)
            landscape = text_to_landscape(NKMODEL_DATAPATH)
            ctrbs = im_landscape_to_contributions(im, landscape, A=args_.A)
        kwag_['objective'] = NK_COMBO(args_.N, args_.K, A=args_.A, im=im, ctrbs=ctrbs,
                                      random_seeds=(case_seed_[0], case_seed_[1], init_seed_))
    else:
        if dir_name_ is None:
            raise NotImplementedError
    
    t = time()
    log_dir, opt_info = COMBO(**kwag_)
    optimum_combo, opt_state_ind, combo_opt_time = opt_info
    combo_total_time = time() - t

    bo_data = torch.load(os.path.join(log_dir, 'bo_data.pt'))
    inputs, outputs = bo_data['eval_inputs'], bo_data['eval_outputs']

    local_optima, _ = torch.cummin(outputs.view(-1), dim=0) # negative valued.
    
    if objective_ == 'nkmodel':
        model = kwag_['objective'].nkmodel
        model.print_info(path=log_dir)
        fit_opt, states_opt, landscape = model.get_global_optimum(cache=True)
        local_gaps = (local_optima + fit_opt).tolist()  # local_optima is negative valued.
        assert len(local_gaps) == args_.n_eval
        bo_data['local_gaps'] = local_gaps
        torch.save(bo_data, os.path.join(log_dir, 'bo_data.pt'))
        if RECORD:
            writer = SummaryWriter(log_dir=log_dir)
            inputs = [tuple(x) for x in inputs.int().tolist()]
            outputs = (-outputs.int().view(-1)).tolist()
            states = sorted(landscape.keys())
            states_strs = ["".join([str(y) for y in x]) for x in states]
            landscape_list = [landscape[x] for x in states]
            assert len(inputs) == len(outputs) == args_.n_eval

            fig1 = plt.figure(figsize=(15,8))
            plt.plot(states_strs, landscape_list)
            plt.xticks(rotation=90)
            plt.scatter(["".join([str(y) for y in x]) for x in states_opt], [fit_opt]*len(states_opt), marker='*', color='tab:orange', s=300)
            plt.title(f"N={args_.N} K={args_.K} (init: {INITIAL_POINTS_N})")
            for i in range(args_.n_eval):
                plt.scatter(["".join([str(y) for y in inputs[i]])], [outputs[i]], marker=f'${i+1}$', color='k', s=200)
            writer.add_figure('Search Order on Fitness Landscape', fig1)

            fig2 = plt.figure(figsize=(10,8))
            plt.plot(list(range(1,args_.n_eval+1)), local_gaps)
            plt.xlabel("iterations (1 ~ n_eval)")
            plt.ylabel("abs(glob_opt - loc_opt)")
            writer.add_figure('Gap btw Global-local optimum', fig2)

            print("Plot of Lanscape and Search steps Completed.")
    
    
    t = time()
    true_inputs = torch.Tensor(list(itertools.product(range(args_.A), repeat=args_.N)))
    true_outputs = kwag_['objective'].evaluate(true_inputs).view(-1,1)
    optimum_naive = true_outputs.min().item()
    opt_state_ind = true_outputs.argmin().item()
    opt_state = true_inputs[opt_state_ind]
    naive_time = time()-t
    assert kwag_['objective'].evaluate(opt_state) == optimum_naive
    print( "=====================================")
    print("* Runtime comparison:")
    print(f"COMBO total run time: {combo_total_time:.4f} sec.")
    print(f"COMBO loc. opt. run time: {combo_opt_time:.4f} sec.")
    print(f"Naive run time: {naive_time:.4f} sec.")
    print("* Result comparison:")
    print(f"COMBO Result: {-optimum_combo}")
    print(f"True Optimum: {-optimum_naive}")

