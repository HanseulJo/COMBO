import os
import glob
import torch
import matplotlib.pyplot as plt
from main_NKmodel import *

#directories = []
DIRECTORY = '.'  # 'Experiments/Run###'
IM_SEED = ''  # ''

filenames = glob.glob(os.path.join(f'{DIRECTORY}/NK_COMBO_{IM_SEED}*', 'bo_data.pt'))
bo_datas = [torch.load(fn) for fn in filenames]

keys = ['local_optima', 'random_cummax', 'fit_opt']
for i in range(len(bo_datas))[::-1]:
    if not all([k in bo_datas[i] for k in keys]):
        del bo_datas[i]
print(f"Using {len(bo_datas)} runs to make a plot...")

optimums = torch.Tensor([-data['fit_opt'] for data in bo_datas]).view(-1,1)
local_optimas = torch.cat([data['local_optima'].view(1,-1) for data in bo_datas])
random_cummaxs = -torch.cat([data['random_cummax'].view(1,-1) for data in bo_datas])
datas_tensor = local_optimas - optimums
random_gap = random_cummaxs - optimums

length = len(datas_tensor[0])
data_means = datas_tensor.mean(axis=0)
data_stds = datas_tensor.std(axis=0)
x = torch.arange(1,length+1)
random_means = random_gap.mean(axis=0)
random_stds = random_gap.std(axis=0)
print("COMBO result (gap):", data_means[-1])
print("random result (gap):", random_means[-1])

plt.plot(x, data_means, linewidth=2, label='COMBO (local search)')
plt.fill_between(x, (data_means - data_stds), (data_means + data_stds), color='tab:blue', alpha=0.2)
plt.plot(x, random_means, linewidth=2, label='Random Walk', color='tab:gray')
plt.fill_between(x, (random_means - random_stds), (random_means + random_stds), color='tab:gray', alpha=0.1)
plt.title('Gap: Global Optimum - Tentative Optima (Mean $\pm$ Std.dev)')
plt.legend()
plt.xticks(range(0, 21, 4))
plt.ylim(-0.05, 0.55)
plt.show()



