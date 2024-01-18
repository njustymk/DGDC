import os
import pdb
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm

import args
import utils
import model
import dataset



def test_one(args, net, group, sample):
    out = net(group.to(args.device), sample.to(args.device))
    out = torch.nn.functional.softmax(out, dim=1)
    score = out[0][0].data.cpu().numpy()
    return score

def test(test_loader, net, args):
    print('test......')
    # net.train()

    loss0_show = 0
    loss1_show = 0
    loss_show = 0
    results = []

    inf_num = 0
    inf_time_all = 0
    # for group_face, group_topk, group_topk_fake, face_normal, face_fevers, topk_normal, topk_fevers, label1, label2, sample_id_face, bbox_face in tqdm(test_loader):
    for group_face, group_topk, face_normal, face_fevers, topk_normal, topk_fevers, label1, label2, sample_id_face, bbox_face in tqdm(test_loader):
        result = []
        # t1 = time.time()
        result.append(test_one(args, net, group_topk, topk_normal))
        # t2 = time.time()
        # inf_time_all = inf_time_all+t2-t1
        # inf_num = inf_num+1
        # if inf_num>100:
        #     print(inf_time_all/inf_num)
        #     break

        for topk_fever in topk_fevers:
            topk_fever = torch.unsqueeze(topk_fever, dim=0)
            result.append(test_one(args, net, group_topk, topk_fever))
        results.append([sample_id_face[0], bbox_face[0].numpy(), result])

    return results


def main():

    epoch = args.test_epoch
    weight_file = os.path.join(args.weights_dir, 'train_split{}-train_s{}-train_e{}-epoch{}.pt'.\
            format(args.train_split, args.train_s, args.train_e, epoch))

    if not os.path.exists(args.results_dir): os.mkdir(args.results_dir)

    # 测试数据队列
    test_set = dataset.Dataset_test(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # 模型
    net = model.DGDC(args=args).to(args.device)
    net.load_state_dict(torch.load(weight_file, map_location='cpu'))

    # 测试
    results = test(test_loader, net, args)
    
    # 评估
    print('evaluate......')
    utils.evaluate_results(args, results)

    # 保存测试结果
    results_file = os.path.join(args.results_dir, 'results.txt')
    print('Results save to: ', results_file)
    utils.write_results(args, results, results_file)



if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    print('time: ', time.asctime())

    args = args.get_args()

    main()