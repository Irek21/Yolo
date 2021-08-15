import torch
import numpy as np

from torch import abs
from torch import pow
from torch import tensor

from geometry import iou
from geometry import iou
from geometry import cnt_box

L_NOOBJ = 0.5
L_COORD = 5

def criterion(predict, y):
    loss = 0
    bsize = predict.shape[0]
    
    bool_grid = np.array([[True for j in range(7)] for i in range(7)])

    for i in range(bsize):
        objs = y[i]
        for obj in objs:

            box_truth = obj['box']
            box1 = cnt_box(predict[i, obj['cell'][0], obj['cell'][1],  :4])
            box2 = cnt_box(predict[i, obj['cell'][0], obj['cell'][1], 4:8])
            bool_grid[obj['cell'][0], obj['cell'][1]] = False

            if iou(*box_truth, *box1) > iou(*box_truth, *box2):

                loss += L_COORD * pow(predict[i, obj['cell'][0], obj['cell'][1], :2] - 
                                      tensor(obj['coord'][:2]).cuda(), 2).sum()

                loss += L_COORD * pow(pow(abs(predict[i, obj['cell'][0], obj['cell'][1], 2:4]), 1/2) - 
                                      pow(tensor(obj['coord'][2:4]).cuda(), 1/2), 2).sum()

                loss += pow(predict[i, obj['cell'][0], obj['cell'][1], 8] - 1, 2)

            else:
                loss += L_COORD * pow(predict[i, obj['cell'][0], obj['cell'][1], 4:6] - 
                                      tensor(obj['coord'][:2]).cuda(), 2).sum()
                loss += L_COORD * pow(pow(abs(predict[i, obj['cell'][0], obj['cell'][1], 6:8]), 1 / 2) - 
                                      pow(tensor(obj['coord'][2:4]).cuda(), 1/2), 2).sum()

                loss += pow(predict[i, obj['cell'][0], obj['cell'][1], 9] - 1, 2)
            
            cl_target = torch.zeros(20).cuda()
            cl_target[obj['cl_num']] = 1.0
            loss += pow(predict[i, obj['cell'][0], obj['cell'][1], 10:] - cl_target, 2).sum()
            
        noobj_confs = predict[i, bool_grid][:, 8:10]
        f_confs = torch.zeros(noobj_confs.shape[0], 2).cuda()
        loss += L_NOOBJ * pow(noobj_confs - f_confs, 2).sum()
        
    return loss / bsize