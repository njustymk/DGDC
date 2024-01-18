import os
import pdb
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def modified_temp(normal, bbox_face, temp_step):
    # temp_step = 0.3
    bbox_h = bbox_face[3]-bbox_face[1]
    distance_face = (1. / (bbox_h+1))
    normal = normal + temp_step * distance_face
    return normal
    
def read_bbox(bbox_file):
    bboxes = []
    f = open(bbox_file, 'r')
    datas = f.readlines()
    for data in datas:
        parts = data.strip().split(' ')
        [xmin, ymin, xmax, ymax, score]= parts
        bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax), float(score)])
    return bboxes

def prepare_data(data_dir):
    temp_faces = []
    sample_ids_face = []
    bboxes_face = []

    bbox_ir_dir = os.path.join(data_dir, 'bbox_ir')

    # 获取当前路径下所有的个体ID
    sample_ids = [x[:18] for x in os.listdir(bbox_ir_dir)]
    sample_ids = np.sort(sample_ids)
    for sample_id in tqdm(sample_ids):

        # 读取温度数据
        raw = np.load(os.path.join(data_dir, 'npy',  sample_id+'.npy'))
        bbox_ir_file = os.path.join(bbox_ir_dir, sample_id+'.txt')
        bboxes = read_bbox(bbox_ir_file)

        for bbox in bboxes:
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # 提取面部信息
            temp_face = raw[ymin:ymax, xmin:xmax].astype(np.float32)
            temp_faces.append(temp_face)

            sample_ids_face.append(sample_id)
            bboxes_face.append(np.array([xmin, ymin, xmax, ymax]))

    data_list = [temp_faces, sample_ids_face, bboxes_face]
    return data_list

def get_data(args, data_split):

    cachefile = os.path.join(args.data_root, 'cache', 'cache_{}.pkl'.format(data_split))
    print(cachefile)
    if os.path.isfile(cachefile):
        print('loading...', cachefile)
        with open(cachefile, "rb") as f:
            data_list = pickle.load(f)
        print('Finish')
    else:
        print('prepare data......')
        data_dir = os.path.join(args.data_root, data_split)
        data_list = prepare_data(data_dir)
        with open(cachefile, "wb") as f:
            pickle.dump(data_list, f)
        print('Finish')
    return data_list

def write_results(args, results, results_file):
    f = open(results_file, 'w')
    for result in results:
        sample_id = result[0]
        bbox = result[1]
        pre = result[2]
        f.write(sample_id)
        for b in bbox:
            f.write(' '+str(b))
        for p in pre:
            f.write(' '+str(p))
        f.write('\n')
    f.close()

def show_results(accs, accs_img_file):
    xs = range(len(accs))
    ys = accs
    plt.clf()
    plt.plot(xs, ys,'ro-',color='blue')
    plt.savefig(accs_img_file)

# # 评估代码
# def evaluate_results(args, results):
#     print('evaluate......')
#     categories = []
#     for result in results:
#         sample_id = result[0]
#         bbox = result[1]
#         pre = result[2]
#         pre = np.array(pre)
#         category = pre[:, 1]
#         categories.append(category)
#     categories = np.array(categories)

#     accs = ''
#     acc = (categories[:,0]==0).sum()/len(categories)
#     acc = format(acc, '.3f')
#     acc = '&'+str(acc)
#     # acc = '0-'+str(acc)
#     accs = accs+acc+' '
#     for i in range(1, args.fever_num+1):
#         acc = (categories[:,i]==1).sum()/len(categories)
#         acc = format(acc, '.3f')
#         acc = '&'+str(acc)
#         # acc = str(i)+'-'+str(acc)
#         accs = accs+acc+' '
#     print(accs)


# 评估代码
def evaluate_results(args, results):
    print('evaluate......')
    roc_aucs = []
    for faver_id in range(args.fever_num):
        y_label = []
        y_score = []
        for result in results:
            sample_id = result[0]
            bbox = result[1]
            pre = result[2]
            y_label.append(1)
            y_score.append(float(pre[0]))

            y_label.append(0)
            y_score.append(float(pre[1+faver_id]))
        fpr, tpr, thersholds = roc_curve(y_label, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_auc = format(roc_auc, '.4f')
        roc_aucs.append(roc_auc)

    print(roc_aucs)
    return roc_auc

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.MLP1 = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#         )

#         self.MLP1 = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#         )

#     def forward(self, x):
#         x1 = self.MLP1(x)
#         x2 = self.MLP2(x)
