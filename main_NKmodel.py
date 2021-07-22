import argparse
import numpy as np
import torch
import os
import itertools
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from time import time

from main import COMBO
from NKmodel import NKmodel
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

def _generate_random_seeds(seed_str, n_im_seed=3, n_ctrbs_seed=3, n_init_point_seed=3):
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_im_seed):
        result[rng_state.randint(0, 10000)] = (list(rng_state.randint(0, 10000, (n_ctrbs_seed,))), list(rng_state.randint(0, 10000, (n_init_point_seed,))))
    return result

def generate_random_seeds_nkmodel():
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    return _generate_random_seeds(seed_str="NK_MODEL", n_im_seed=100, n_ctrbs_seed=100, n_init_point_seed=100)

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

class NK_COMBO(object):
    """
    Preprocessing NK model to solve it with COMBO
    """
    def __init__(self, N, K, A=2, im=None, ctrbs=None, random_seeds=(None, None, None), start_from_bottom=False):
        self.n_vertices = np.repeat(A, N)
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
        if start_from_bottom:
            anti_opt_list = self.nkmodel.get_optimum_and_more(INITIAL_POINTS_N, anti_opt=True)
            anti_opt_states, ind = [], 0
            while len(anti_opt_states) < INITIAL_POINTS_N:
                anti_opt_states += anti_opt_list[ind]["states"]
                ind += 1
            anti_opt_states = anti_opt_states[:INITIAL_POINTS_N]
            self.suggested_init = torch.cat([torch.Tensor(state).long().view(1,-1) for state in anti_opt_states])
        else:
            self.suggested_init = sample_init_points(self.n_vertices, INITIAL_POINTS_N, random_seed=random_seeds[-1])

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

def random_wide_search(states, inputs, landscape, args):
    random_input_inds = np.random.choice(np.arange(len(states)), size=args.n_eval-INITIAL_POINTS_N, replace=False)
    random_inputs = inputs[:INITIAL_POINTS_N] + [states[i] for i in random_input_inds]
    #print("random_inputs:\n", random_inputs)
    random_outputs = [landscape[state] for state in random_inputs]
    random_cummax, _ = torch.cummax(torch.Tensor(random_outputs), dim=0)
    return random_cummax

def random_local_search(states, inputs, landscape, args):
    random_input_loci = np.random.choice(np.arange(args.N), size=args.n_eval - INITIAL_POINTS_N)
    curr_state = list(inputs[np.argmin([landscape[inputs[j]] for j in range(INITIAL_POINTS_N)]).item()])
    random_inputs = inputs[:INITIAL_POINTS_N]  # Use the same Initial Points
    for locus in random_input_loci:
        k = curr_state[locus]
        curr_state[locus] = 1-k
        random_inputs.append(tuple(curr_state))
    #print("random_inputs:\n", random_inputs)
    random_outputs = [landscape[state] for state in random_inputs]
    random_cummax, _ = torch.cummax(torch.Tensor(random_outputs), dim=0)
    return random_cummax

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='COMBO : Combinatorial Bayesian Optimization using the graph Cartesian product')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=1)
    parser_.add_argument('--N', dest='N', type=int, default=6)
    parser_.add_argument('--K', dest='K', type=int, default=1)
    parser_.add_argument('--A', dest='A', type=int, default=2)
    parser_.add_argument('--dir_name', dest='dir_name')
    parser_.add_argument('--objective', dest='objective', default='nkmodel')
    parser_.add_argument('--parallel', dest='parallel', action='store_true', default=False)
    parser_.add_argument('--device', dest='device', type=int, default=None)
    parser_.add_argument('--task', dest='task', type=str, default='both')
    parser_.add_argument('--game_num', dest='game_num', type=int, default=None)
    parser_.add_argument('--interdependency_seed', dest='interdependency_seed', type=int, default=None)
    parser_.add_argument('--payoff_seed', dest='payoff_seed', type=int, default=None)
    parser_.add_argument('--init_point_seed', dest='init_point_seed', type=int, default=None)
    parser_.add_argument('--start_from_bottom', dest='start_from_bottom', action='store_true', default=False)
    parser_.add_argument('--local_search', dest='local_search', action='store_true', default=False)

    args_ = parser_.parse_args()

    if args_.game_num is not None:
        USE_DATA = True
        GAME_NUM = args_.game_num
        NKMODEL_DATAPATH = f"../Exp input/Game{GAME_NUM} landscape.txt"
        NKMODEL_IM_PATH = f"../Exp input/Game{GAME_NUM} knowledge.txt"

    print(args_)
    kwag_ = vars(args_)
    dir_name_ = kwag_['dir_name']
    objective_ = kwag_['objective']
    parallel_ = kwag_['parallel']
    if args_.interdependency_seed is None:
        kwag_['interdependency_seed'] = np.random.randint(0,100)
    if args_.payoff_seed is None:
        kwag_['payoff_seed'] = np.random.randint(0,100)
    if args_.init_point_seed is None:
        kwag_['init_point_seed'] = np.random.randint(0,100)
    seed_info = (kwag_['interdependency_seed'],kwag_['payoff_seed'],kwag_['init_point_seed'])
    if args_.device is None:
        del kwag_['device']
    print(kwag_)
    assert (dir_name_ is None) != (objective_ is None)

    if objective_ == 'nkmodel':
        random_seeds = generate_random_seeds_nkmodel()
        im_seed_ = sorted(random_seeds.keys())[seed_info[0]]
        ctrbs_seed_list_, init_seed_list_ = sorted(random_seeds[im_seed_])
        ctrbs_seed_ = ctrbs_seed_list_[seed_info[1]]
        init_seed_ = init_seed_list_[seed_info[2]]

        manual_seed = False
        if manual_seed:
            im_seed_ = 371
            ctrbs_seed_ = 2174
            init_seed_ = 1092

        im, ctrbs = None, None
        if USE_DATA:
            if NKMODEL_IM_PATH is not None:
                im = text_to_interdependence(NKMODEL_IM_PATH)
            if NKMODEL_DATAPATH is not None:
                landscape = text_to_landscape(NKMODEL_DATAPATH)
            ctrbs = im_landscape_to_contributions(im, landscape, A=args_.A)
        kwag_['objective'] = NK_COMBO(args_.N, args_.K, A=args_.A, im=im, ctrbs=ctrbs,
                                      random_seeds=(im_seed_, ctrbs_seed_, init_seed_),
                                      start_from_bottom=args_.start_from_bottom)
    else:
        if dir_name_ is None:
            raise NotImplementedError
    
    t = time()
    log_dir, opt_info = COMBO(**kwag_)
    optimum_combo, opt_state_ind, combo_opt_time = opt_info
    combo_total_time = time() - t

    bo_data = torch.load(os.path.join(log_dir, 'bo_data.pt'))
    inputs, outputs = bo_data['eval_inputs'], bo_data['eval_outputs']

    local_optima, _ = torch.cummin(outputs.view(-1), dim=0)
    assert len(local_optima) == args_.n_eval
    
    if objective_ == 'nkmodel':
        bo_data['local_optima'] = local_optima  # save it as a negative-valued tensor.
        local_optima = -local_optima # flip the sign: positive valued.

        model = kwag_['objective'].nkmodel
        model.print_info(path=log_dir)
        fit_opt, states_opt, landscape = model.get_global_optimum(cache=True)
        bo_data['fit_opt'] = fit_opt
        if RECORD:
            writer = SummaryWriter(log_dir=log_dir)
            inputs = [tuple(x) for x in inputs.int().tolist()]
            #print("COMBO_inputs:\n", inputs)
            outputs = -outputs.view(-1) # positive valued.
            states = sorted(landscape.keys())
            states_strs = ["".join([str(y) for y in x]) for x in states]
            landscape_list = [landscape[x] for x in states]
            assert len(inputs) == len(outputs) == args_.n_eval

            # Random Search (to Compare with Evaluation Plot)
            if kwag_['local_search']:
                random_cummax = random_local_search(states, inputs, landscape, args_)
            else:
                random_cummax = random_wide_search(states, inputs, landscape, args_)  # or, random_local_search
            bo_data['random_cummax'] = random_cummax
            
            # Plot 1: Landscpe and Searching order
            fig1 = plt.figure()
            plt.plot(states_strs, landscape_list, label='Landscape')
            plt.xticks(rotation=90)
            plt.scatter(["".join([str(y) for y in x]) for x in states_opt], [fit_opt]*len(states_opt),
                        marker='*', color='tab:orange', s=300, label='Global optimum')
            plt.title(f"N={args_.N} K={args_.K} (init: {INITIAL_POINTS_N})")
            for i in range(args_.n_eval):
                plt.scatter(["".join([str(y) for y in inputs[i]])], [outputs[i]], marker=f'${i+1}$', color='k', s=200)
            plt.legend()
            plt.xlabel("states (00...0 ~ 11...1)")
            plt.ylabel("fitness values")
            writer.add_figure('Search Order on Fitness Landscape', fig1)

            # Plot 2: Evaluation Plot
            fig2 = plt.figure()
            x = list(range(1,args_.n_eval+1))
            plt.plot(x, outputs, label='Evaluations')
            plt.plot(x, local_optima, linewidth=5, color='r', label='Optimum so far')
            plt.axhline(y = fit_opt, color='tab:orange', linestyle='--', label='Global optimum')
            plt.plot(x, random_cummax, color='tab:gray', label='Random Cumul. Optim')
            plt.xlabel("iterations (1 ~ n_eval)")
            plt.ylabel("local optima")
            plt.title(f"N={args_.N} K={args_.K} (init={INITIAL_POINTS_N}, interdep_num={seed_info[0]})")
            plt.legend()
            writer.add_figure('Evaluation Plot', fig2)
            print("Plot of Landscape and Search steps Completed.")
        
    torch.save(bo_data, os.path.join(log_dir, 'bo_data.pt'))
    
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


# graph kernel?