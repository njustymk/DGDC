import os
import gc
import sys
import cv2
import pdb
# import math
import torch
import pickle
import numpy as np
import torchvision
from tqdm import tqdm
from scipy.linalg import expm,logm

import utils

def modified_temp(normal, bbox_face, temp_step):
    # temp_step = 0.3
    bbox_h = bbox_face[3]-bbox_face[1]
    distance_face = (1. / (bbox_h+1))
    normal = normal + temp_step * distance_face
    return normal

def trans_face(face, args):
    face[face<args.temp_min]=args.temp_min
    face[face>args.temp_max]=args.temp_max
    face = cv2.resize(face, (args.face_w, args.face_h), cv2.INTER_LINEAR)
    face = face.astype(np.float32)
    face = (face - args.temp_min)/(args.temp_max - args.temp_min)
    face = np.exp(face)
    face = face*255.
    return face

def get_topk(face, args):
    topk = -np.sort(-face.flatten())
    topk = topk[args.vector_s :args.vector_s +args.vector_len]
    return topk

def get_face_ids(normals, args):    
    face_ids = []
    for i in range(args.len_old+args.end_old, len(normals), args.len_sampling):
        face_ids.append(i)
    return face_ids

class Dataset_train(torch.utils.data.Dataset):
    def __init__(self, args):
        super(Dataset_train, self).__init__()
        self.args = args

        data_list = utils.get_data(args, args.train_split)
        self.normals         = data_list[ 0][args.train_s: args.train_e]
        self.sample_ids_face = data_list[-2][args.train_s: args.train_e]
        self.bboxes_face     = data_list[-1][args.train_s: args.train_e]

        self.face_ids = get_face_ids(self.normals, self.args)
        
    def get_group(self, face_id):
        group_face = []
        group_topk = []
        for i in range(face_id -self.args.len_old-self.args.end_old, face_id-self.args.end_old):
            face = self.normals[i].copy()
            face = modified_temp(face, self.bboxes_face[face_id], self.args.temp_step)
            face = trans_face(face, self.args)
            topk = get_topk(face, self.args)
            group_face.append(face)
            group_topk.append(topk)
        group_face = np.array(group_face)
        group_topk = np.array(group_topk)
        return group_face, group_topk

    # def get_subject(self, face_id):
    #     face_normal = self.normals[face_id].copy()
    #     face_normal = modified_temp(face_normal, self.bboxes_face[face_id], self.args.temp_step)
    #     face_fever1 = face_normal + self.args.fever1_train
    #     face_fever2 = face_normal + self.args.fever2_train

    #     face_normal = trans_face(face_normal, self.args)
    #     face_fever1 = trans_face(face_fever1, self.args)
    #     face_fever2 = trans_face(face_fever2, self.args)

    #     topk_normal = get_topk(face_normal, self.args)
    #     topk_fever1 = get_topk(face_fever1, self.args)
    #     topk_fever2 = get_topk(face_fever2, self.args)

    #     face_normal = np.array([face_normal])
    #     face_fever1 = np.array([face_fever1]) 
    #     face_fever2 = np.array([face_fever2])

    #     topk_normal = np.array([topk_normal])
    #     topk_fever1 = np.array([topk_fever1]) 
    #     topk_fever2 = np.array([topk_fever2])

    #     return face_normal, face_fever1, face_fever2, topk_normal, topk_fever1, topk_fever2

    def get_subject(self, face_id):
        face = self.normals[face_id]
        face = modified_temp(face, self.bboxes_face[face_id], self.args.temp_step)

        face_normal = trans_face(face, self.args)
        topk_normal = get_topk(face_normal, self.args)

        face_fevers = []
        topk_fevers = []
        for i in range(1, self.args.fever_num+1):
            face_fever = face +self.args.fever_gap *i
            face_fever = trans_face(face_fever, self.args)
            topk_fever = get_topk(face_fever, self.args)
            face_fevers.append(face_fever)
            topk_fevers.append(topk_fever)

        return face_normal, face_fevers, topk_normal, topk_fevers

    def __len__(self):
        return len(self.face_ids)

    def __getitem__(self, index):

        face_id = self.face_ids[index]

        group_face, group_topk = self.get_group(face_id)
        # face_normal, face_fever1, face_fever2, topk_normal, topk_fever1, topk_fever2 = self.get_subject(face_id)
        face_normal, face_fevers, topk_normal, topk_fevers = self.get_subject(face_id)

        label0 = 0
        label1 = 1

        sample_id_face = self.sample_ids_face[face_id]
        bbox_face = self.bboxes_face[face_id]
        
        # return group_face, group_topk, face_normal, face_fever1, face_fever2, topk_normal, topk_fever1, topk_fever2, \
        #         label0, label1, sample_id_face, bbox_face

        return group_face, group_topk, face_normal, face_fevers, topk_normal, topk_fevers, label0, label1, sample_id_face, bbox_face

class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, args):
        super(Dataset_test, self).__init__()
        self.args = args

        data_list = utils.get_data(args, args.test_split)
        self.normals         = data_list[ 0][args.test_s: args.test_e]
        self.sample_ids_face = data_list[-2][args.test_s: args.test_e]
        self.bboxes_face     = data_list[-1][args.test_s: args.test_e]

        self.face_ids = get_face_ids(self.normals, self.args)

    def get_group(self, face_id):
        group_face = []
        group_topk = []
        for i in range(face_id -self.args.len_old-self.args.end_old, face_id-self.args.end_old):
            face = self.normals[i].copy() +self.args.temp_offset
            face = modified_temp(face, self.bboxes_face[face_id], self.args.temp_step)
            face = trans_face(face, self.args)
            topk = get_topk(face, self.args)
            group_face.append(face)
            group_topk.append(topk)

        return group_face, group_topk

    def get_subject(self, face_id):
        face = self.normals[face_id]
        face = modified_temp(face, self.bboxes_face[face_id], self.args.temp_step)

        face_normal = trans_face(face, self.args)
        topk_normal = get_topk(face_normal, self.args)

        face_fevers = []
        topk_fevers = []
        for i in range(1, self.args.fever_num+1):
            face_fever = face +self.args.fever_gap *i
            face_fever = trans_face(face_fever, self.args)
            topk_fever = get_topk(face_fever, self.args)
            face_fevers.append(face_fever)
            topk_fevers.append(topk_fever)

        return face_normal, face_fevers, topk_normal, topk_fevers

    def __len__(self):
        return len(self.face_ids)

    def __getitem__(self, index):

        face_id = self.face_ids[index]

        group_face, group_topk = self.get_group(face_id)


        face_normal, face_fevers, topk_normal, topk_fevers = self.get_subject(face_id)
        face_normal = np.array([face_normal])
        topk_normal = np.array([topk_normal])

        label1 = 0
        label2 = 1

        sample_id_face = self.sample_ids_face[face_id]
        bbox_face = self.bboxes_face[face_id]

        group_topk_fake = group_topk.copy()
        group_topk_fake[np.random.randint(self.args.len_old-self.args.end_old)] = topk_fevers[-1]
        # group_topk_fake[np.random.randint(self.args.len_old-self.args.end_old)] = topk_fevers[-1]
        # group_topk_fake[np.random.randint(self.args.len_old-self.args.end_old)] = topk_fevers[-1]
        # group_topk_fake[np.random.randint(self.args.len_old-self.args.end_old)] = topk_fevers[-1]
        # group_topk_fake[np.random.randint(self.args.len_old-self.args.end_old)] = topk_fevers[-1]



        group_face = np.array(group_face)
        group_topk = np.array(group_topk)
        group_topk_fake = np.array(group_topk_fake)

        # return group_face, group_topk, group_topk_fake, face_normal, face_fevers, topk_normal, topk_fevers, label1, label2, sample_id_face, bbox_face
        return group_face, group_topk, face_normal, face_fevers, topk_normal, topk_fevers, label1, label2, sample_id_face, bbox_face

if __name__ == '__main__':
    data_root = 'data/data_raw/20200808'
    dataset = Dataset(data_root)
