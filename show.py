import os
import cv2
import pdb
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt

import iron

def draw_board(face_list, categories_list):
    board = np.zeros((300, 1600, 3), np.uint8)
    # redline = np.zeros((5, 128, 3), np.uint8)
    # greenline = 
    for index, face in enumerate(face_list):
        categories = categories_list[index]
        board[0:128, 135*index:135*index+128, :] = face

        for cat_id in range(len(categories)):
            board[128+15*cat_id :128+15*cat_id+10, 135*index:135*index+128, categories[cat_id]+1] = 255
    return board


def main():
    border = 20
    scene = 'scene2'
    method = 'threshold-fix'
    results_dir = os.path.join(data_dir, method, 'results/summer')

    face_list = []
    categories_list = []
    npy_dir = os.path.join(data_dir, 'data', 'data_FFIR', scene, 'npy')
    bbox_dir = os.path.join(data_dir, 'data', 'data_FFIR', scene, 'bbox_ir')
    results_file = os.path.join(results_dir, 'results.txt')
    

    with open(results_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        sample_id_face = line[0]
        print(sample_id_face)
        xmin = float(line[1])
        ymin = float(line[2])
        xmax = float(line[3])
        ymax = float(line[4])
        # categories = [int(x) for x in line[5:]]
        # print(categories)
        # categories_list.append(categories)

        npy_file = os.path.join(npy_dir, sample_id_face+'.npy')
        bbox_file = os.path.join(bbox_dir, sample_id_face+'.txt')

        raw = np.load(npy_file)
        raw_face = raw[int(ymin):int(ymax), int(xmin):int(xmax)]

        img = (raw-raw.min())*255/(raw.max()-raw.min())
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        img_face = img[int(ymin)-border:int(ymax)+border, int(xmin)-border:int(xmax)+border]
        img_face = cv2.resize(img_face, (128, 128))

        face_list.append(img_face)
        # board = draw_board(face_list, categories_list)

        # if(len(face_list))>10:
        #     del face_list[0]
        #     del categories_list[0]

        cv2.imshow('img', img)
        cv2.imshow('img_face', img_face)
        # cv2.imshow('board', board)
        if cv2.waitKey()==113: break   # 'q'
        
        # pdb.set_trace()


if __name__ == '__main__':
    # data_dir = '../data/data_FFIR/scene2'
    data_dir = 'Z:/code/fever_screening'
    methods = [
    'threshold-fix', 
    'threshold-dynamic', 
    # 'cnn',
    'dgdc-max', 
    'dgdc-img', 
    'dgdc-topk', 
    'dgdc-cnn', 
    'dgdc',
    ]
    
    main()

