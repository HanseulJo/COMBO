import numpy as np

import torch

from COMBO.acquisition.acquisition_optimizers.graph_utils import neighbors
from COMBO.acquisition.acquisition_marginalization import acquisition_expectation
from COMBO.acquisition.acquisition_functions import expected_improvement

N_RANDOM_VERTICES = 20000
N_GREEDY_ASCENT_INIT = 20
N_SPRAY = 20


def optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                acquisition_func=expected_improvement, reference=None, do_local_search=False):
    """
    :param x_opt: 1D Tensor
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices:
    :param acquisition_func:
    :param reference:
    :return:
    """
    N_RANDOM_VERTICES = min(2**len(x_opt), 20000) # If we use (N=6,K=1) model, Huge value of N_RANDOM_VERTICES is redundant.
    if do_local_search:
        min_nbd = neighbors(x_opt, partition_samples, edge_mat_samples, n_vertices, uniquely=True)
        shuffled_ind = list(range(len(x_opt)))
        x_init_candidates = min_nbd[shuffled_ind]
    else:
        rnd_nbd = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(N_RANDOM_VERTICES, 1)) for n_v in n_vertices]), dim=1).long()
        min_nbd = neighbors(x_opt, partition_samples, edge_mat_samples, n_vertices, uniquely=False)
        shuffled_ind = list(range(min_nbd.size(0)))
        np.random.shuffle(shuffled_ind)
        x_init_candidates = torch.cat(tuple([min_nbd[shuffled_ind[:N_SPRAY]], rnd_nbd]), dim=0)
    
    acquisition_values = acquisition_expectation(x_init_candidates, inference_samples, partition_samples, n_vertices,
                                                 acquisition_func, reference)

    nonnan_ind = ~torch.isnan(acquisition_values).squeeze(1)
    x_init_candidates = x_init_candidates[nonnan_ind]
    acquisition_values = acquisition_values[nonnan_ind]

    acquisition_sorted, acquisition_sort_ind = torch.sort(acquisition_values.squeeze(1), descending=True)
    x_init_candidates = x_init_candidates[acquisition_sort_ind]

    return x_init_candidates[:N_GREEDY_ASCENT_INIT], acquisition_sorted[:N_GREEDY_ASCENT_INIT]


