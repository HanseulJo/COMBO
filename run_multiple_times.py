import os

start_from_bottom = False
local_search = True

command = lambda x: f"python ./main_NKmodel.py --N 6 --K 1 --A 2 --n_eval 20 --interdependency_seed {x[0]} --payoff_seed {x[1]} {'--start_from_bottom' if start_from_bottom else ''} {'--local_search' if local_search else ''}"

for i in range(10):
    for j in range(10):
        os.system(command([i,j]))

