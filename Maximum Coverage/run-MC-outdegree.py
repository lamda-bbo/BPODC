import multiprocessing
import os
import argparse

def func(name, times_index):
    cmd=f'python MC-outdegree-main.py -alg {name}  -times {times_index}'
    print(cmd)
    os.system(cmd)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-times_min',  type=int, default=5)
    argparser.add_argument('-times_max',  type=int, default=29)

    args = argparser.parse_args()

    algos=["DY_POMC","DY_EAMC","DY_FPOMC","DY_BPODC"]

    pool = multiprocessing.Pool(processes=30*len(algos))
    for alg in algos:
        for times in range(args.times_min, args.times_max+1,1):
            pool.apply_async(func, (alg,times,))

    pool.close()
    pool.join()

