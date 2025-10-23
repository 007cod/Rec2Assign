import numpy as np
import time
import os 
import pandas as pd
import random
import torch
from utils import Worker, Task, action_vis, write_data
from assignment import CBTR, worker_selection
from dataloader import loaddata
np.random.seed(0)
random.seed(0)

import cProfile
import pstats

def main(args):
    
    workers, tasks, max_time, fact = loaddata(args)
    STmodel = CBTR(args)
    
    ti, wi = 0, 0
    vis_datas, reward = [], []
    start_time = time.time()
    for i in range(0, int(max_time + 1000), 10): 
        newtasks = []
        newworkers = []
        while ti < len(tasks) and tasks[ti].publish_time <= i:
            newtasks.append(tasks[ti])
            ti += 1
        while wi < len(workers) and workers[wi].publish_time <= i:
            newworkers.append(workers[wi])
            wi += 1
        if STmodel.init(i, newtasks, newworkers) != -1:
            STmodel.recommenda()
            if args["recom_method"] == "OTA":
                selec_candidate = STmodel.workers_candidate
            else:
                selec_candidate = worker_selection(STmodel.c_tasks, STmodel.c_workers, max_selec_num = args['max_selec_num'], t = i)
            results = STmodel.assign(selec_candidate)
            STmodel.update(results)
        vis_datas.append(STmodel.get_vis_data(i))
        reward.append(STmodel.opt)
        print("time: {}, profit:{:.2f}".format(i, STmodel.opt))
    print("Time cost: {:.2f}".format(time.time()-start_time))
    
    methods_name = args["recom_method"]
    mean_success, fairness = STmodel.get_fairness_score()
    
    write_data(path="/data/chenjinwen/cjw/CompeteRecommendSC/output/"+args["data_name"]+"/"+args["different"]+".txt", 
               data={
                   args["different"]:args[args["different"]],
                   "method":methods_name,
                   "Time":time.time()-start_time,
                   "profit":STmodel.opt,
                   "sucess_ration":mean_success,
                   "fairness_score":fairness,}
               )
    # action_vis(vis_datas, reward)
    

if __name__ == "__main__":
    args = {
    "data_name":"dd",  #dd, yc
    "task_num":5000,
    "worker_num":600,
    "wdead":2,
    "sdead":0.5,
    "wait_time":30,
    "range":3,
    "profit_fact":2,
    "v":25/3600,
    "k":2,
    "max_selec_num":2,
    "is_demand":False,
    "soon_to_free":False,
    "fairness": False,
    "recom_method":"MKM",
    "different":"task_num",
    "device":'cuda:2' if torch.cuda.is_available() else 'cpu',
    }

    main(args)

    