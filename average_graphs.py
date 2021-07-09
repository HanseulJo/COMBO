import os
import torch
import matplotlib.pyplot as plt
from main_NKmodel import *

directories = [
    # add directories of experiments
]
filenames = [os.path.join(d, 'bo_data.pt') for d in directories]
bo_datas = [torch.load(fn) for fn in filenames]

datas = [data['local_gaps'] for data in bo_datas]
length = len(datas[0])
assert all([len(data)==length for data in datas])
datas_np = torch.Tensor(datas)
data_means = datas_np.mean(axis=0)
data_stds = datas_np.std(axis=0)
x = torch.arange(1,length+1)

plt.plot(x, data_means, linewidth=2)
plt.fill_between(x, (data_means - data_stds), (data_means + data_stds), color='r', alpha=0.1)
plt.show()



