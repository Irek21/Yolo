import torch
import numpy as np

names = ['person', 
         'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
         'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 
         'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
nums = np.arange(20)
classes = {names[i]:nums[i] for i in range(20)}

def to_target(anns, predict):
    bsize = len(anns)
    target = [[] for i in range(bsize)]
    for i in range(bsize):
        
        width  = int(anns[i]['size']['width'])
        height = int(anns[i]['size']['height'])
        
        for j in range(len(anns[i]['object'])):
            
            xmax = int(anns[i]['object'][j]['bndbox']['xmax']) / width
            xmin = int(anns[i]['object'][j]['bndbox']['xmin']) / width
            ymax = int(anns[i]['object'][j]['bndbox']['ymax']) / height
            ymin = int(anns[i]['object'][j]['bndbox']['ymin']) / height
                        
            xcenter = (xmax + xmin) / 2
            ycenter = (ymax + ymin) / 2
            
            cell = (int(xcenter / (1 / 7)),
                    int(ycenter / (1 / 7)))
            
            box = (xmax, xmin, ymax, ymin)
            coord = (xcenter, ycenter, xmax - xmin, ymax - ymin)
            
            name = anns[i]['object'][j]['name']
            cl_num = classes[name]
            
            obj = {
                'cell': cell,
                'coord': coord,
                'box': box,
                'cl_num': cl_num
            }
            
            target[i].append(obj)
    return target

def iou(xmax1, xmin1, ymax1, ymin1, xmax2, xmin2, ymax2, ymin2):
    s_inter = max((min(xmax1, xmax2) - max(xmin1, xmin2)) * (min(ymax1, ymax2) - max(ymin1, ymin2)), 0)
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    return s_inter / (s1 + s2 - s_inter)

def cnt_box(coord_arr):
    xcenter = coord_arr[0]
    ycenter = coord_arr[1]
    w = coord_arr[2]
    h = coord_arr[3]
    
    xmax = xcenter + (w / 2)
    xmin = xcenter - (w / 2)
    
    ymax = ycenter + (h / 2)
    ymin = ycenter - (h / 2)
    
    return (xmax, xmin, ymax, ymin)