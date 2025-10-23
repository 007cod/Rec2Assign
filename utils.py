import math
import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import os


import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mu=30, sigma=2, lower=5, upper=np.inf, size=1):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)[0]

class Task():
    def __init__(self, id, publish_time, x, y, ex, ey, d, r, wait_time, dead_time, **kwarg) -> None:
        self.id = id
        self.publish_time = publish_time
        self.end_time = publish_time + dead_time
        self.wait_time = wait_time
        self.x = x
        self.y = y
        self.ex = ex
        self.ey = ey
        self.r = r
        self.d = math.sqrt(pow(ex-x, 2) + pow(ey-y, 2))
        
        self.recommended_worker_list = []
        # self.cx = x
        # self.cy = y
        self.is_fail = False
        self.is_do = False
        self.is_finish = False
        self.__dict__.update(kwarg)
            

class Worker():
    def __init__(self, id, publish_time, x, y, fact, v, dead_time, **kwarg) -> None:
        self.id = id
        self.publish_time = publish_time
        self.x = x
        self.y = y
        self.v = v
        
        self.bx = x
        self.by = y
        self.is_task = False
        self.current_task = None
        self.last_tasks = []
        self.end_time = publish_time + dead_time
        self.reject_task_ids_list = []
        self.recommend_task_list = []
        
        D = (random.random()*1)/111*fact
        self.labda = -math.log(0.5)/D
        self.p = 1
        self.sucess_num = 0
        self.fail_num = 0
        
        self.idel_time = -1
        
        self.__dict__.update(kwarg)
        
def worker_selection(tasks:list[Task], workers:list[Worker], max_selec_num, t):
    out_candidate = []
    for i, w in enumerate(workers):
        temp, prob = [], []
        if w.idel_time ==-1:
            w.idel_time = truncated_normal() + t
        elif t >= w.idel_time:
            if len(w.recommend_task_list) ==0:
                out_candidate.append(temp)
                continue
            for s in w.recommend_task_list:
                
                dis = math.sqrt(pow(w.x-s.x, 2) + pow(w.y-s.y, 2)) + 1e-8
                p = math.exp(-w.labda*dis)
                prob.append([s, p])
            prob = sorted(prob, key=lambda x:x[1], reverse=True)
            for s, p in prob:
                s_idx = tasks.index(s)
                if len(temp) >= max_selec_num:
                    break
                if random.random() <=p:
                    temp.append(s_idx) 
                # else:
                #     if s.id in w.reject_task_ids_list:
                #         print("error")
                #     w.reject_task_ids_list.append(s.id)
                
            w.idel_time =-1
            
        out_candidate.append(temp)
    return out_candidate 


from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

def action_vis(data, reward):
    fig, ax = plt.subplots()

    # 三类散点
    sc_worker = ax.scatter([], [], marker='o', color='blue')
    sc_task = ax.scatter([], [], marker='*', color='red')
    task_pred_end = ax.scatter([], [], marker='D', color='green')

    # 新增：任务和预测之间的连线集合
    lines = LineCollection([], colors='gray', linewidths=1, alpha=0.6)
    ax.add_collection(lines)

    # 文本
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black')
    reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='black')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    def update(frame):
        if len(data[frame]):
            x, y, c = zip(*data[frame])
        else:
            x, y, c = [], [], []

        # 各类点
        task_points = np.column_stack(([d for i, d in enumerate(x) if c[i]==0 ],
                                       [d for i, d in enumerate(y) if c[i]==0 ]))
        pred_points = np.column_stack(([d for i, d in enumerate(x) if c[i]==2 ],
                                       [d for i, d in enumerate(y) if c[i]==2 ]))

        sc_worker.set_offsets(np.column_stack(([d for i, d in enumerate(x) if c[i]==1 ],
                                               [d for i, d in enumerate(y) if c[i]==1 ])))
        sc_task.set_offsets(task_points)
        task_pred_end.set_offsets(pred_points)

        #按下标对应一一相连
        segs = []
        for i in range(min(len(task_points), len(pred_points))):
            segs.append([task_points[i], pred_points[i]])
        lines.set_segments(segs)

        # 文本
        time_text.set_text(f'Time: {frame}')
        reward_text.set_text(f'R: {reward[frame]}')

        return sc_worker, sc_task, task_pred_end, lines, time_text, reward_text

    ani = FuncAnimation(fig, update, frames=range(len(data)), interval=100, blit=True)
    ani.save("animation.gif", writer=PillowWriter(fps=25))

    # 显示动画
    # plt.show()
    
def write_data(path, data):
    f = open(path, "+a")
    # line = " ".join(f"{k}:{v:.2f}" for k, v in data.items())
    text = []
    for k, v in data.items():
        if isinstance(v, float):
            text.append(f"{k}:{v:.3f}")
        else:
            text.append(f"{k}:{v}")
    line = " ".join(text)
    f.write(line + "\n")