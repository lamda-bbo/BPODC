**This package includes the Python code of the paper 'Biased Pareto Optimization for Subset Selection with Dynamic Cost Constraints'.** 

ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Dr. Chao Qian (qianc@lamda.nju.edu.cn).

ATTN2: This package was developed by Ms. Dan-Xuan Liu (liudx@lamda.nju.edu.cn). For any problem concerning the code, please feel free to contact Ms. Dan-Xuan Liu.



## Required packages:

  1. `numpy`
  2. `tqdm`
  3. `matplotlib`
  4. `multiprocessing`

## Run Influence Maximization Task:

 - `IM-outdegree-main.py` includes all the code necessary for running influence maximization tasks, such as loading data, reading sequences of dynamic budget changes, invoking algorithmic optimization, and writing logs. You can view the experimental settings at the "main entry point."
 - `run-IM-outdegree.py` utilizes multiprocessing to parallelize the execution of optimization algorithms on another script, `IM-outdegree-main.py`. 
 - `/outdegree` datasets are saved in this folder.


## Run Maximum Coverage Task:

 - `MC-outdegree-main.py` includes all the code necessary for running maximum coverage tasks, such as loading data, reading sequences of dynamic budget changes, invoking algorithmic optimization, and writing logs. You can view the experimental settings at the "main entry point."
 - `run-MC-outdegree.py` utilizes multiprocessing to parallelize the execution of optimization algorithms on another script, `MC-outdegree-main.py`. 
 - `/outdegree` datasets are saved in this folder.

