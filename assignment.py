import numpy as np
import os 
import pandas as pd
import sys
from utils import Task, Worker, worker_selection
import copy
import torch
import torch.optim as optim
import math
import random
from collections import defaultdict
from itertools import combinations
from scipy.optimize import linear_sum_assignment

class CBTR():
    def __init__(self, args, **kwarg) -> None:
        self.args = args
        self.t = 0
        self.tasks = []
        self.workers = []
        
        self.c_tasks = []
        self.c_workers = []
        self.do_worker_id = []
        self.STB_worker_id = []
        
        self.k = self.args['k']
        self.opt = 0
        
        self.is_demand = self.args['is_demand']
        self.__dict__.update(kwarg)
        
    
    def init(self, t, tasks:list[Task], workers:list[Worker]):
        self.t = t
        self.tasks += tasks
        self.workers += workers
        
        self.c_tasks += tasks
        self.c_workers += workers
        
        next_do_worker_id, next_STB_worker_id = [], []
        for w_id in self.do_worker_id:
            w = self.workers[w_id]
            self.update_w_pos(w)
            s = w.current_task
            if w.end_time > self.t and w.finish_time <= self.t:
                self.c_workers.append(w)
                self.opt += w.this_r
                s.is_finish = True
                w.is_task = False
                w.current_task = None
                w.last_tasks.append(s.id)
            else:
                next_do_worker_id.append(w_id)
                if w.finish_time - self.t < 20:
                    next_STB_worker_id.append(w_id)
        self.do_worker_id = next_do_worker_id
        self.STB_worker_id = next_STB_worker_id
        self.worker_num = len(self.c_workers)
        self.task_num = len(self.c_tasks)
        self.worker_selection = [[] for _ in range(len(self.c_workers))]
        
        if self.worker_num == 0:
            self.update([])
            return -1
        elif self.task_num == 0:
            self.update([[] for _ in range(len(self.c_workers))])
            return -1

    def update(self, results):
        next_c_workers = []
        next_c_tasks = []
        for i, w in enumerate(self.c_workers):
            temp_recommend = []
            for s in w.recommend_task_list:
                if s.id in w.reject_task_ids_list:
                    s.recommended_worker_list.remove(w)
                else:
                    temp_recommend.append(s)
            w.recommend_task_list = temp_recommend
            if len(results[i]) > 0:
                s = self.c_tasks[results[i][0]]
                w.is_task = True
                w.current_task = s
                w.begin_time = self.t
                w.finish_time = self.t + (self.distance(w, s) + math.sqrt(pow(s.ex-s.x, 2) + pow(s.ey-s.y, 2)) ) // w.v
                w.this_r = s.r - self.args["profit_fact"]*(self.distance(w, s) + s.d)
                w.bx, w.by = w.x, w.y
                s.is_do = True
                self.do_worker_id.append(w.id)
                w.sucess_num += 1
                for w2 in s.recommended_worker_list:
                    w2.recommend_task_list.remove(s)
                    
                for s2 in w.recommend_task_list:
                    s2.recommended_worker_list.remove(w)
                    
                w.recommend_task_list = []
                s.recommended_worker_list = []
                
                if w.idel_time != -1:
                    print("false")
            else:
                next_c_workers.append(w)
                if len(self.worker_selection[i]) > 0:
                    w.fail_num += 1
        
        for s in self.c_tasks:
            if s.publish_time + s.wait_time < self.t:
                s.is_fail = True
            if s.is_do == False and s.is_fail == False:
                next_c_tasks.append(s)
        
        self.c_tasks = next_c_tasks
        self.c_workers = next_c_workers
        
        for w in self.c_workers:
            w.recommend_task_list = [s for s in w.recommend_task_list if not (s.is_fail or s.is_do)]
            
                    
    def get_fairness_score(self):
        score_list = []
        n = len(self.workers)
        for w in self.workers:
            if w.sucess_num + w.fail_num == 0:
                n -= 1 
                # score_list.append(0)
            else:
                score_list.append(w.sucess_num / (w.sucess_num + w.fail_num))
        
        # mean_success = sum(score_list[i] for i in range(n))/n
        # fairness = 1 - sum(abs(score_list[i] - score_list[j]) for i in range(len(score_list)) for j in range(i+1, len(score_list)))/((n-1) * n+ 1e-8)
        
        if len(score_list) == 0:
            return 0, 0  # 无有效工人时返回 0
        
        mean_success = sum(score_list) / len(score_list)
        # 使用 Jain's Fairness Index
        sum_x = sum(score_list)
        sum_x2 = sum(x**2 for x in score_list)
        fairness = (sum_x ** 2) / (len(score_list) * sum_x2) if sum_x2 > 0 else 0
        
        return mean_success, fairness
        
    def update_w_pos(self, w):
        s = w.current_task
        w.x, w.y = self.worker_position(w, s, self.t- w.begin_time)

    def worker_position(self, w, s, t):
        # 距离
        d_ws = math.hypot(s.x - w.bx, s.y - w.by)
        d_se = math.hypot(s.ex - s.x, s.ey - s.y)

        # 时间
        t_ws = d_ws / w.v if w. v > 0 else float('inf')
        t_se = d_se / w.v if w.v > 0 else float('inf')

        # 阶段1：工人前往任务点
        if t < t_ws:
            ratio = t / t_ws
            x = w.bx + (s.x - w.bx) * ratio
            y = w.by + (s.y - w.by) * ratio
            return x, y

        # 阶段2：执行任务
        elif t < t_ws + t_se:
            ratio = (t - t_ws) / t_se
            x = s.x + (s.ex - s.x) * ratio
            y = s.y + (s.ey - s.y) * ratio
            return x, y

        # 阶段3：任务完成
        else:
            return s.ex, s.ey
        
    def get_RS(self, worker_to):
        self.dis = np.zeros((self.worker_num, self.task_num))
        self.p = np.zeros((self.worker_num, self.task_num))
        self.bonus = np.ones((self.worker_num, self.task_num))
                    
        for i, w in enumerate(self.c_workers):
            for j, s in enumerate(self.c_tasks):
                self.dis[i][j] = self.distance(w, s)
                base_p = math.exp(-w.labda*self.dis[i][j])
                # 软偏好：如果 worker_to 建议该工人去某格子，则给予 bonus（不再把其它位置全部设 p=0）
                if self.is_demand and worker_to is not None:
                    pref = worker_to[i]  # a tuple or None
                    task_cell = (s.x, s.y)
                    if pref is not None:
                        # if pref == task_cell:
                        #     self.bonus[i][j] =  1.2  # 完全匹配的显著奖励
                        # else:
                        #     # 如果 pref 与 task_cell 距离较近，也给小奖励
                        #     dd = math.hypot(pref[0] - task_cell[0], pref[1] - task_cell[1])
                        #     if dd <= 1:
                        #         self.bonus[i][j]  = 1.1
                        #     elif dd <= 2:
                        #         self.bonus[i][j]  =  1.05
                        dd =  math.sqrt(pow(pref[0]-task_cell[0], 2) + pow(pref[1]-task_cell[1], 2)) + 1e-8
                        self.bonus[i][j]  =  1 + 0.2 * max(0, (2 - dd))
                        # print(self.bonus[i][j])
                    self.p[i][j] = min(1, base_p * self.bonus[i][j])
                else:
                    self.p[i][j] = base_p

                # 保持 reject 列表逻辑
                if s.id in w.reject_task_ids_list:
                    self.dis[i][j] = 1e2
                    self.p[i][j] = 0
                    
        tasks_candidate, edges  = [], []
        for i, s in enumerate(self.c_tasks):
            temp_nearest_idx = []
            # if len(s.recommended_worker_list) < self.k:
            # if True:
            nearest_idx = np.argsort(self.dis[:,i])
            for j in nearest_idx:
                if self.dis[j][i] > 1e5:
                    break
                if (self.dis[j][i] + math.sqrt(pow(s.ex-s.x, 2) + pow(s.ey-s.y, 2))) / self.c_workers[j].v + self.t > s.end_time:
                    continue
                temp_nearest_idx.append(j)
                edges.append([j, i, (s.r - self.args["profit_fact"]*(self.dis[j][i] + s.d)) / (self.dis[j][i] + s.d)])
            tasks_candidate.append(temp_nearest_idx)
            
        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        return tasks_candidate, edges
    
    
    
    def reduce_edge(self, candidate, edges):            
        for i, j, _ in edges:
            w = self.c_workers[i]
            s = self.c_tasks[j]
            # if len(s.recommended_worker_list) >=self.k:  # or w.prob() < 0.05
            if len(s.recommended_worker_list) >= (self.t - s.publish_time)//5 :
                    continue
            if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
            w.recommend_task_list.append(s)
            s.recommended_worker_list.append(w)
                
    def near_reco(self, ):
        for i, w in enumerate(self.c_workers):
            nearest_idx = np.argsort(self.dis[i,:])
            for j in nearest_idx[:self.k]:
                s = self.c_tasks[j]
                if self.dis[i][j] > 1e5:
                        break
                if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
                w.recommend_task_list.append(s)
                s.recommended_worker_list.append(w)
    
    def no_compete(self, candidate):
        for j, s in enumerate(self.c_tasks):
            for i in candidate[j]:
                w = self.c_workers[i]
                if len(s.recommended_worker_list) >=1:
                    break
                if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
                w.recommend_task_list.append(s)
                s.recommended_worker_list.append(w)
    
    def OTA(self, candidate):
        self.workers_candidate = [[] for _ in range(self.worker_num)]
        for i in range(self.task_num):
            for j in candidate[i]:
                self.workers_candidate[j].append(i)
        for i, w in enumerate(self.c_workers):
            for s_i in self.workers_candidate[i]:
                s = self.c_tasks[s_i]
                if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
                w.recommend_task_list.append(s)
                s.recommended_worker_list.append(w)
    
    def multi_KM(self, candidate):
        self.workers_candidate = [[] for _ in range(self.worker_num)]
        for i in range(self.task_num):
            for j in candidate[i]:
                self.workers_candidate[j].append(i)
        base_profits = np.zeros((self.worker_num, self.task_num))
        for i in range(self.worker_num):
            for j in self.workers_candidate[i]:
                base_profits[i][j] = (self.c_tasks[j].r - self.args["profit_fact"]*(self.dis[i][j] + self.c_tasks[j].d)) / (self.dis[i][j] + self.c_tasks[j].d) * self.p[i][j]
                # if self.is_demand:
                #     base_profits[i][j] *= self.bonus[i][j]
        
        temp_p = np.zeros((self.worker_num, self.task_num))
        profits = copy.deepcopy(base_profits)
        out_candidate = [[] for _ in range(self.worker_num)]
        
        for i in range(self.worker_num):
            for s in self.c_workers[i].recommend_task_list:
                idx = self.c_tasks.index(s)
                base_profits[i][idx] = 0
                profits[i][idx] = 0
                temp_p[i][idx] = self.p[i][idx]
                out_candidate[i].append(idx)
        
        for _ in range(10):
            row_ind, col_ind = linear_sum_assignment(profits, maximize=True)
            for idx, w_i in enumerate(row_ind):
                if base_profits[w_i][col_ind[idx]] != 0:
                    out_candidate[w_i].append(col_ind[idx])
                    base_profits[w_i][col_ind[idx]] = 0
                    temp_p[w_i][col_ind[idx]] = self.p[w_i][col_ind[idx]]
            
            #normlize
            sum_w_p = np.sum(temp_p, axis=1)
            g = temp_p / (np.sum(temp_p, axis=1, keepdims=True) + 1e-8) * ( 1 - np.cumprod((1 - temp_p), axis=1))
            w_prob =  1 - np.prod((1 - temp_p), axis=1)
            
            prob_worker_reject_all_task = 1 - np.sum(g, axis=1)
            
            # g = g / (np.sum(g, axis=0, keepdims=True) + 1e-8) * ( 1 - np.cumprod((1 - g), axis=0))
            
            sum_s_p = np.sum(g, axis=0)
            
            prob_task_no_select  = np.prod((1 - g), axis=0)
                    
            for s_j in range(self.task_num):
                for i in candidate[s_j]:
                    if prob_worker_reject_all_task[i].item() * prob_task_no_select[s_j].item()  < 0.01:
                        profits[i][s_j] = 0
                    else:
                        profits[i][s_j] = base_profits[i][s_j] * prob_worker_reject_all_task[i].item() * prob_task_no_select[s_j].item()
            
            if np.sum(profits) < 10:
                break
            
            
        for i in range(self.worker_num):
            for j in out_candidate[i]:
                w = self.c_workers[i]
                s = self.c_tasks[j]
                if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
                w.recommend_task_list.append(s)
                s.recommended_worker_list.append(w)
    
    def DP(self, O_SP, w_prob, w_i, s_j, p, sum_s, profit):
        new_O_SP = copy.deepcopy(O_SP)
        new_O_SP[s_j] = p
        new_w_prob = 1 - np.cumprod((1 - new_O_SP))
        
        Delt = O_SP * (w_prob - new_w_prob) / np.sum(O_SP)
        
        return np.sum(Delt/sum_s * profit)
    
    def CRA(self, candidate):
        def gain(temp_s, pref_matrix, s_i, w_i, idx): 
            if idx < self.args["k"]:
                last = temp_s[s_i] / (1 - pref_matrix[w_i][s_i])
                return temp_s[s_i] - last
            else:
                new = temp_s[s_i] * (1 - pref_matrix[w_i][s_i])
                return new - temp_s[s_i]
        
        self.workers_candidate = [[] for _ in range(self.worker_num)]
        for i in range(self.task_num):
            for j in candidate[i]:
                self.workers_candidate[j].append(i)
                
        self.temp_s = np.ones((len(self.c_tasks)))
        for w_i, w in enumerate(self.c_workers):
            for s_i in self.workers_candidate[w_i][:self.args["k"]]:
                self.temp_s[s_i] *= (1 - self.p[w_i][s_i]+0.1)
        #get max profit
        iter_num = 0
        flag = True
        while(iter_num < 3 and flag):
            iter_num += 1
            flag = False
            for w_i, w in enumerate(self.c_workers):
                temp_gain = []
                for idx, s_i in enumerate(self.workers_candidate[w_i]):
                    temp_gain.append(gain(self.temp_s, self.p, s_i, w_i, idx))
                new_RS = [x for _, x in sorted(zip(temp_gain, self.workers_candidate[w_i]))]
                for idx, s_i in enumerate(new_RS):
                    if idx < self.args["k"] and s_i not in self.workers_candidate[w_i][:self.args["k"]]:
                        self.temp_s[s_i] *= (1 - self.p[w_i][s_i] + 0.1)
                        flag = True
                    elif idx >= self.args["k"] and s_i in self.workers_candidate[w_i][:self.args["k"]]:
                        self.temp_s[s_i] /= (1 - self.p[w_i][s_i] + 0.1)
                        flag = True
                self.workers_candidate[w_i] = new_RS

        for i in range(self.worker_num):
            for j in self.workers_candidate[i][:self.args["k"]]:
                w = self.c_workers[i]
                s = self.c_tasks[j]
                if len(w.recommend_task_list) >= self.args["k"]:
                    break
                if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
                    continue
                w.recommend_task_list.append(s)
                s.recommended_worker_list.append(w)
                
                
    # def GCA(self, candidate):
    #     lenR = [[0, len(x), i] for i, x in enumerate(candidate)]
    #     num = 0
    #     while num < self.worker_num * self.k:
    #         lenR = sorted(lenR, key=lambda x: (x[0], x[1]))
    #         s_i = lenR[0][2]
    #         if lenR[0][1] == 1e9:
    #             break
    #         flag = 0
    #         for id, w_i in enumerate(candidate[s_i]):
    #             w = self.c_workers[w_i]
    #             s = self.c_tasks[s_i]
    #             if len(w.recommend_task_list) < self.k:
    #                 if s.id in w.reject_task_ids_list or s in w.recommend_task_list:
    #                     continue
    #                 w.recommend_task_list.append(s)
    #                 s.recommended_worker_list.append(w)
    #                 lenR[0][1] -=1
    #                 lenR[0][0] +=1
    #                 candidate[s_i].pop(id)
    #                 flag = 1
    #                 num +=1
    #                 break
    #         if flag == 0:
    #             lenR[0][1] = 1e9
    #             candidate[s_i] = []
                
    def GCA(self, candidate):
        lenR = [[len(x), i] for i, x in enumerate(candidate)]
        num = 0
        while num < self.worker_num * self.k:
            lenR = sorted(lenR, key=lambda x: x[0])
            s_i = lenR[0][1]
            if lenR[0][0] == 1e9:
                break
            flag = 0
            for id, w_i in enumerate(candidate[s_i]):
                w = self.c_workers[w_i]
                s = self.c_tasks[s_i]
                if len(w.recommend_task_list) < self.k:
                    if s.id in w.reject_task_ids_list or s in w.recommend_task_list or len(self.c_tasks[s_i].recommended_worker_list) >= 1:
                        continue
                    w.recommend_task_list.append(s)
                    s.recommended_worker_list.append(w)
                    lenR[0][0]  = 1e9
                    candidate[s_i].pop(id)
                    flag = 1
                    num +=1
                    break
            if flag == 0:
                lenR[0][0] = 1e9
                candidate[s_i] = []
    
    def DYTOPK(self, candidate):
        from scipy.spatial.distance import cdist
        self.workers_candidate = [[] for _ in range(self.worker_num)]
        for i in range(self.task_num):
            for j in candidate[i]:
                self.workers_candidate[j].append(i)
                

        coords = np.array([[w.x, w.y] for w in self.c_workers])  # shape: (N, 2)

        disw = cdist(coords, coords, metric='euclidean')  # 自动向量化，快！

        np.fill_diagonal(disw, np.inf)
        k = min(3, self.worker_num)
        nearest_indices = np.argpartition(disw, kth=k-1, axis=1)[:, :k]  # shape: (N, 3)

        candidate_lengths = np.array([len(self.workers_candidate[i]) for i in range(self.worker_num)])
        worker_k = (candidate_lengths[nearest_indices].sum(axis=1) // 3).tolist()
                
                
        # worker_k = [0 for _ in range(self.worker_num)]
        # disw = np.zeros((self.worker_num, self.worker_num))
        
        # for w_i in range(self.worker_num):
        #     for w_j in range(self.worker_num):
        #         wi = self.c_workers[w_i]
        #         wj = self.c_workers[w_j]
        #         if w_i != w_j:
        #             disw[w_i][w_j] = math.sqrt(pow(wi.x-wj.x, 2) + pow(wi.y-wj.y, 2)) + 1e-8
        #         else:
        #             disw[w_i][w_j] = float('inf')  # Avoid self-comparison
        #     nearest_indices = np.argpartition(disw[w_i], kth=min(2, len(disw[w_i])-1))[:3]
        #     worker_k[w_i] = sum(len(self.workers_candidate[i]) for i in nearest_indices)//3
            
        tasks = self.c_tasks  # alias for readability
        max_task_workers = math.ceil(self.worker_num * self.k / self.task_num)
        changed = True
        iter_num = 3
        while changed and iter_num > 0:
            changed = False
            iter_num -= 1
            for w_i, worker in enumerate(self.c_workers):
                if len(worker.recommend_task_list) >= worker_k[w_i]:
                    continue

                candidates = []
                for task_id in self.workers_candidate[w_i]:
                    task = tasks[task_id]
                    if len(task.recommended_worker_list) >= max_task_workers:
                        continue  # 跳过已满的任务（p 视为 0）
                    p_val = self.p[w_i][task_id]
                    if p_val > 0:
                        candidates.append((p_val, task))

                if not candidates:
                    continue

                best_p, best_task = max(candidates, key=lambda x: x[0])
                if best_task.id in worker.reject_task_ids_list or best_task in worker.recommend_task_list:
                    continue

                worker.recommend_task_list.append(best_task)
                best_task.recommended_worker_list.append(worker)
                changed = True
                
            
    def distance(self, w, s):
        return math.sqrt(pow(w.x-s.x, 2) + pow(w.y-s.y, 2)) + 1e-8
    
    def recommenda(self, ):
        if self.is_demand:
            worker_to = self.get_supply_demand_worker_to()
        else:
            worker_to = None
        self.tasks_candidate, self.edges = self.get_RS(worker_to)
        
        if self.args["recom_method"]=="NEA":
            self.near_reco()
        elif self.args["recom_method"]=="NOC":
            self.no_compete(self.tasks_candidate)
        elif self.args["recom_method"]=="RL":
            self.RL(self.tasks_candidate)
        elif self.args["recom_method"]=="GCA":
            self.GCA(self.tasks_candidate)
        elif self.args["recom_method"]=="OTA":
            self.OTA(self.tasks_candidate)
        elif self.args["recom_method"].find("MKM") != -1:
            self.multi_KM(self.tasks_candidate)
        elif self.args["recom_method"]=="CRA":
            self.CRA(self.tasks_candidate)
        elif self.args["recom_method"]=="DYTOPK":
            self.DYTOPK(self.tasks_candidate)   
        else:
            self.reduce_edge(self.tasks_candidate, self.edges)

        # org_candidate = [[self.c_tasks[j].id for j in candidate[i]] for i in range(self.worker_num)]
        return 
    
    def KM_assign(self, candidate):
        
        succ_rates = [w.sucess_num / (w.sucess_num + w.fail_num + 1e-8) for w in self.c_workers]
        mean_succ = np.mean(succ_rates) if self.worker_num > 0 else 0.5
        
        profits = np.zeros((self.worker_num, self.task_num))
        for i in range(self.worker_num):
            boost = 1.0
            if self.args["fairness"]:
                if succ_rates[i] != 0:
                    boost *= math.exp(mean_succ - succ_rates[i])
                
            for j in candidate[i]:
                w = self.c_workers[i]
                d = self.dis[i][j] + self.c_tasks[j].d
                profits[i][j] = (self.c_tasks[j].r - self.args["profit_fact"]*d) / d * boost

        row_ind, col_ind = linear_sum_assignment(profits, maximize=True)
        candidate = [[] for _ in range(self.worker_num)]
        
        for idx, w_i in enumerate(row_ind):
            if profits[w_i][col_ind[idx]]:
                candidate[w_i].append(col_ind[idx])
        
        return candidate
    
    def assign(self, worker_selection):
        self.worker_selection = worker_selection
        candidate = self.KM_assign(worker_selection)
        results = [[candidate[i][0]] if len(candidate[i]) else [] for i in range(self.worker_num)]
        
        return results
    
    def get_vis_data(self, t):
        vis_data = []
        for s in self.tasks:
            if s.is_finish or s.is_fail:
                continue
            vis_data.append([s.x, s.y, 0])
            vis_data.append([s.ex, s.ey, 2])
        for w in self.workers:
            vis_data.append([w.x, w.y, 1])
        return vis_data

    def get_supply_demand_worker_to(self, map_size=5, max_pick_radius=3):
        """
        更宽松的 supply-demand -> worker_to 映射：
        - 仍用 MCF 得到每个 (ori_cell -> dest_cell, num) 的转移量
        - 但在把转移量映射到具体 worker 时，不仅限于 ori_cell 内的工人，
        而是从全局可用工人列表中按距离 dest_cell 排序挑选 nearest workers。
        - 参数 max_pick_radius：最多考虑多远范围的工人（格子距离/欧式距离尺度，可调）
        """
        s_map = np.zeros((map_size, map_size))
        soon_to_free_workers = np.zeros((map_size, map_size))
        # 统计需求
        for s in self.c_tasks:
            if not s.is_do and not s.is_fail:
                s_map[int(s.x), int(s.y)] += 1
        # 考虑即将释放的工人
        for w in self.workers:
            if w.is_task and w.finish_time - self.t < 10:
                s_map[int(w.current_task.ex), int(w.current_task.ey)] -= 1
                soon_to_free_workers[int(w.current_task.ex), int(w.current_task.ey)] += 1
        # 减去当前空闲工人的所在格子（它们已算作供给）
        for w in self.c_workers:
            s_map[int(w.x), int(w.y)] -= 1

        points_de, points_add = [], []
        for i in range(map_size):
            for j in range(map_size):
                if s_map[i, j] > 0:
                    points_add.append([(i, j), int(s_map[i, j]), []])   # 需要补充的点
                elif s_map[i, j] < 0:
                    points_de.append([(i, j), int(s_map[i, j]), []])    # 过剩的点（可迁出）

        # map_worker: 把所有空闲（或soon-to-free）工人扁平化为一个列表供后续选择
        map_worker = {}  # 原来按格子分，现在仍保留方便统计
        flat_workers = []  # [ [worker_idx, x, y, available_time_flag] , ... ]
        for i, w in enumerate(self.c_workers):
            coord = (int(w.x), int(w.y))
            if coord not in map_worker:
                map_worker[coord] = []
            map_worker[coord].append([i, w.x, w.y])
            flat_workers.append([i, w.x, w.y])

        # Build MCF (跟你原来的逻辑类似)
        n = len(points_de) + len(points_add) + 2
        if n <= 2:
            # no flow needed
            worker_to = [(int(w.x), int(w.y)) for w in self.c_workers]
            return worker_to

        MCF = MinCostMaxFlow(n)
        for i, p in enumerate(points_de):
            MCF.add_edge(0, i + 1, -p[1], 0)
            for j, g in enumerate(points_add):
                MCF.add_edge(i + 1, j + 1 + len(points_de), min(g[1], -p[1]), math.hypot(g[0][0] - p[0][0], g[0][1] - p[0][1]))
        for i, g in enumerate(points_add):
            MCF.add_edge(i + 1 + len(points_de), n - 1, g[1], 0)

        flow, cost, ans_flow = MCF.min_cost_max_flow(0, n - 1, len(points_de))

        # 初始化：默认目的地就是当前坐标（不强制移动）
        worker_to = [(int(w.x), int(w.y)) for w in self.c_workers]

        # 扁平化可选工人集合（我们会从中 remove 已分配的工人）
        available_workers = {w[0]: (w[1], w[2]) for w in flat_workers}  # idx -> (x,y)

        # 对每条流，按目标点选择最近的 available workers（不局限于源格子）
        for fl in ans_flow:
            ori_idx = fl[0] - 1
            dest_idx = fl[1] - 1 - len(points_de)
            trans_num = int(fl[2])
            if dest_idx < 0 or ori_idx < 0 or dest_idx >= len(points_add) or ori_idx >= len(points_de):
                continue
            dest_point = points_add[dest_idx][0]
            # 选最近的 trans_num 个可用 worker（全局搜索）
            dist_list = []
            for wid, (wx, wy) in available_workers.items():
                d = math.hypot(wx - dest_point[0], wy - dest_point[1])
                dist_list.append((d, wid))
            dist_list.sort(key=lambda x: x[0])
            picked = [wid for _, wid in dist_list[:trans_num]]
            for wid in picked:
                # 只在 reasonable 半径内才真的移动；否则保留原位（避免把远处人拉过来）
                wx, wy = available_workers[wid]
                if math.hypot(wx - dest_point[0], wy - dest_point[1]) <= max_pick_radius:
                    worker_to[wid] = dest_point
                else:
                    # 如果太远，可以选择不强制移动：把worker_to保持为原位或设为 None
                    worker_to[wid] = (int(wx), int(wy))
                # 从 available_workers 中移除
                del available_workers[wid]

        return worker_to


from collections import deque

class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.edges = []              # (u, v, cap, cost)
        self.graph = [[] for _ in range(n)]  # 邻接表存边索引

    def add_edge(self, u, v, cap, cost):
        # 正向边
        self.edges.append([u, v, cap, cost])
        self.graph[u].append(len(self.edges) - 1)
        # 反向边（初始容量 0，费用取反）
        self.edges.append([v, u, 0, -cost])
        self.graph[v].append(len(self.edges) - 1)

    def spfa(self, s, t):
        """SPFA 求解 s->t 的最短路"""
        n = self.n
        dist = [float('inf')] * n
        inq = [False] * n
        pre = [-1] * n   # 记录前驱边

        dist[s] = 0
        q = deque([s])

        while q:
            u = q.popleft()
            inq[u] = False
            for idx in self.graph[u]:
                u, v, cap, cost = self.edges[idx]
                if cap > 0 and dist[v] > dist[u] + cost + 1e-8:
                    dist[v] = dist[u] + cost
                    pre[v] = idx
                    if not inq[v]:
                        q.append(v)
                        inq[v] = True
        return dist, pre

    def min_cost_max_flow(self, s, t, wn):
        flow, cost = 0, 0
        while True:
            dist, pre = self.spfa(s, t)
            if dist[t] == float('inf'):  # 找不到增广路
                break

            # 找最小可增广流量
            f = float('inf')
            v = t
            while v != s:
                idx = pre[v]
                f = min(f, self.edges[idx][2])
                v = self.edges[idx][0]
                # print(self.edges[idx])

            # 增广
            v = t
            while v != s:
                idx = pre[v]
                self.edges[idx][2] -= f
                self.edges[idx ^ 1][2] += f  # 反向边
                v = self.edges[idx][0]

            flow += f
            cost += f * dist[t]
        
        ans_flow = []
        for i in range(1, wn+1):
            for ei in self.graph[i]:
                e = self.edges[ei]
                if e[1] != 0:
                    ef = self.edges[ei ^ 1]
                    if ef[2]!=0:
                        ans_flow.append([i, e[1], ef[2]])
        
        return flow, cost, ans_flow