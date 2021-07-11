import os
import glob
import torch
import matplotlib.pyplot as plt
from main_NKmodel import *

#directories = []
DIRECTORY = '.'  # 'Run###'
IM_SEED = ''  # ''

filenames = glob.glob(os.path.join(f'{DIRECTORY}/NK_COMBO_{IM_SEED}*', 'bo_data.pt'))
bo_datas = [torch.load(fn) for fn in filenames]

keys = ['eval_outputs', 'local_optima', 'random_cummax']
for i in range(len(bo_datas))[::-1]:
    if not all([k in bo_datas[i] for k in keys]):
        del bo_datas[i]

optimums = torch.cat([data['eval_outputs'].min().view(1,1) for data in bo_datas])
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

plt.plot(x, data_means, linewidth=2, label='COMBO')
plt.fill_between(x, (data_means - data_stds), (data_means + data_stds), color='tab:blue', alpha=0.2)
plt.plot(x, random_means, linewidth=2, label='Random Search', color='tab:gray')
plt.fill_between(x, (random_means - random_stds), (random_means + random_stds), color='tab:gray', alpha=0.1)
plt.title('Gap: Global Optimum - Tentative Optima (Mean $\pm$ Std.dev)')
plt.legend()
plt.xticks(range(0, 21, 4))
plt.show()



