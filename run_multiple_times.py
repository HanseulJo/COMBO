from main_NKmodel import im_landscape_to_contributions
import subprocess

start_from_bottom = True
local_search = True

command = lambda x: f"python ./main_NKmodel.py --N 6 --K 1 --A 2 --n_eval 20 --interdependency_seed {x[0]} --payoff_seed {x[1]} {'--start_from_bottom' if start_from_bottom else ''} {'--local_search' if local_search else ''}"

error_count = 0
n_im = 10
n_ct = 10
for i in range(n_im):
    for j in range(n_ct):
        try:
            subprocess.run([command([i,j])], shell=True, check=True)
        except:
            error_count += 1

print(f"    {error_count}/{n_im*n_ct} Errors Occured.   ")