import numpy as np
from time import sleep


def _improve(curr, prev):
    if curr == prev:
        return "."
    elif curr > prev:
        return f"+{curr-prev:.3f}"
    else:
        return f"-{prev-curr:.3f}"
    
def _show_best(best):
    print("*** Displaying Your Best Attempt So Far ***")
    for key in best:
        print("best", key, ":", best[key])

def _yes_or_no():
    """
    example:
    if _yes_or_no():
        pass
    """
    YORN = 'undefined'
    while YORN.lower()[0] not in ['y', 'n']:
        YORN = input("yes[y] or no[n]: ").strip()
    return YORN.lower()[0] == 'y'

def game(N=6, K=1, chance=18, can_restart=True):
    """AVAILABLE only for model.A==2 (binary valued NK model)"""
    chance = min(chance, 2**(N-1))
    model = NKmodel(N, K, A=2)
    optimum, _ = model.get_global_optimum()
    optlist = model.get_optimum_and_more(min(10, 2**N))

    start = True
    while start:
        print(f"\nNK MODEL GAME MODE ON\n")
        print(f"Before start, write {N} numbers: each number should be 0 or 1.")
        print(f"Separate numbers by ' '(spacebar).")
        INPUT = []
        while len(INPUT) != N or not set(INPUT).issubset(set(['0', '1'])):
            INPUT = input("* Your input: ").strip().split()
            if len(INPUT) != N:
                print(f"Wrong length: write {N} numbers and separate with spacebars.")
            if not set(INPUT).issubset(set(['0', '1'])):
                print("Wrong number: write 0 or 1 only")
        prev_state = [int(x) for x in INPUT]
        prev_fitness, prev_ctrbs = model.fitness_and_contributions(tuple(prev_state))
        previous = {
            "state": prev_state,
            "fitness": prev_fitness,
            "contrib": prev_ctrbs,
            "improve_fitness": _improve(prev_state, prev_state),
            "improve_contrib": [_improve(0, 0) for _ in range(N)]
        }
        best = {
            "state": prev_state,
            "fitness": prev_fitness,
            "round": 0,
        }
        print("Your initial fitness value:", previous["fitness"], '\n')
        
        for ROUND in range(1, chance+1):
            sleep(1)
            print(f"*** Round {ROUND} ***")
            print( "Previous results:")
            print("PREV state:", previous["state"], '<== IMPROVED' if ROUND>1 and best["state"] == previous["state"] else '')
            print("PREV fitness:", previous["fitness"], "({})".format(previous["improve_fitness"]))
            print("PREV improvement of contributions:", previous["improve_contrib"])
            
            print( "NOW: Which digit do U wanna flip?")
            print(f"     Write a number m if you want to flip m-th digit: (1 <= m <= {N})")
            print(f"     Or, Write 'BEST' if you wnat to display the BEST results so far")
            confirm = False
            while not confirm:
                flip = None
                first = True
                while flip not in list(range(1, N+1)):
                    if first:
                        first = False
                    elif INPUT != 'BEST':
                        print("!!!! Wrong input format! Write again: !!!!")
                    INPUT = input("* Your input: ").strip()
                    if INPUT == 'BEST':
                        _show_best(best)
                    elif INPUT.isdigit():
                        flip = int(INPUT)
                flip -= 1  # index: 0 ~ N-1
                state_temp = previous["state"][:]
                state_temp[flip] = 1-state_temp[flip]
                print("Do you really want to change the state as follows?:")
                print(previous["state"], "-->", state_temp)
                if _yes_or_no():
                    confirm = True
                    previous["state"] = state_temp
                else:
                    print("You answered NO: Re-write your input. ")
                    print( "NOW: Which digit do U wanna flip?")
                    print(f"     Write a number m if you want to flip m-th digit: (1 <= m <= {N})")
                    print(f"     Or, Write 'BEST' if you wnat to display the BEST results so far")
            new_state = np.array(previous["state"])
            depend_on_flip = model.interdependence[:,flip]
            new_fitness = previous["fitness"] * N
            improve_ctrbs = ['.' for _ in range(N)]
            for j in np.nonzero(depend_on_flip)[0]:
                ctrb = model.calculate_ith_contribution(new_state, j)
                prev_ctrb = previous["contrib"][j]
                new_fitness = new_fitness - prev_ctrb + ctrb
                previous["contrib"][j] = ctrb
                improve_ctrbs[j] = _improve(ctrb, prev_ctrb)
            new_fitness /= N
            previous["improve_contrib"] = improve_ctrbs
            previous["improve_fitness"] = _improve(new_fitness, previous["fitness"])
            previous["fitness"] = new_fitness
            if best["fitness"] < new_fitness:
                best["fitness"] = new_fitness
                best["state"] = previous["state"]
                best["round"] = ROUND
            print()
        
        print("THE END: All the chances Ran out!")
        print("Finally, your best attempt is:")
        _show_best(best)

        # Assesment
        if abs(best["fitness"] - optimum) <= 1e-4:
            print("\nGreat! You made it!")
            start = False
        else:
            print("\n  Well, everyone has their bad days.\n")
        
        start = start and can_restart
        sleep(3)
        if start:
            print("Do you want to try again? (with the same landscape)")
            if _yes_or_no():
                print("Note: your best score will be deleted.\n")
            else:
                start = False
        if not start:
            print("Do you want to see the score board?")
            if _yes_or_no():
                for i in range(min(10, 2**N)):
                    opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                    print(f"{i+1}-th optimum: {opt} {optstates}")
    
    

if __name__ == "__main__":
    print("Preparing the game.....")
    from main_NKmodel import *
    game(N=6, K=1, chance=18, can_restart=False)

