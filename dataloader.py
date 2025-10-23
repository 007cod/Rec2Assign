import torch
import torch.nn as nn
import numpy as np
import random
from utils import Task, Worker
import os
np.random.seed(42)     
def loaddata(args):
    tasks = [] 
    workers = []
    path = f"/data/chenjinwen/cjw/CompeteRecommendSC/dataset/{args['data_name']}/{args['data_name']}.txt"
    max_time = 0
    with open(path, 'r') as f:
        content = f.readlines()
        head = content[0].split(',')
        fact = float(head[2].split(':')[1])
        for d in content[1:]:
            d = d.split()
            if d[1] != "t":
                workers.append(np.array([float(x) for x in d[0:1] + d[2:]]))
            else:
                tasks.append(np.array([float(x) for x in d[0:1] + d[2:]]))#[t, x, y, ex, ey, tr, v]
                max_time = max(max_time, float(d[0]))
                
    worker_idx_path = f"/data/chenjinwen/cjw/CompeteRecommendSC/dataset/{args['data_name']}/{args['data_name']}_{args['worker_num']}_worker.txt"
    if not os.path.exists(worker_idx_path):
        worker_idx = list(random.sample(range(0, len(workers)), args["worker_num"]))
        worker_idx.sort()
        with open(worker_idx_path, 'w') as f:
            f.write(','.join([str(i) for i in worker_idx]))
    else:
        with open(worker_idx_path, 'r') as f:
            worker_idx = f.read().split(',')
            worker_idx = [int(i) for i in worker_idx]
            
    task_idx_path = f"/data/chenjinwen/cjw/CompeteRecommendSC/dataset/{args['data_name']}/{args['data_name']}_{args['task_num']}_task.txt"
    if not os.path.exists(task_idx_path):
        task_idx = list(random.sample(range(0, len(tasks)), args['task_num']))
        task_idx.sort()
        with open(task_idx_path, 'w') as f:
            f.write(','.join([str(i) for i in task_idx]))
    else:
        with open(task_idx_path, 'r') as f:
            task_idx = f.read().split(',')
            task_idx = [int(i) for i in task_idx]

    workers = [workers[i] for i in worker_idx]
    tasks = [tasks[i] for i in task_idx]
    
    if args['data_name'] == "gMission":
        fact = 1
        out_tasks = [Task(i, *s[0:3], s[1], s[2], 0, s[4],  wait_time=args["wait_time"]) for i, s in enumerate(tasks)]
        out_workers = [Worker(i, *s[0:3], fact=fact, v = args["v"]/111*fact,) for i, s in enumerate(workers)]  
    else:
        out_tasks = [Task(i, *s[0:7], wait_time=args["wait_time"], dead_time=args["sdead"]*3600) for i, s in enumerate(tasks)]
        out_workers = [Worker(i, *s[0:3], fact=fact, v = args["v"]/111*fact, dead_time=args["wdead"]*3600) for i, s in enumerate(workers)]  
    
    return  out_workers, out_tasks, max_time, fact