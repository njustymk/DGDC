import os
import pdb
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm

import args
import model
import utils
import dataset

def train(train_loader, net, optimizer, lossfunc, args):
    print('train......')
    net.train()

    loss0_show = 0
    loss1_show = 0
    loss2_show = 0
    loss_show = 0

    for group_face, group_topk, face_normal, face_fevers, topk_normal, topk_fevers, label0, label1, sample_id_face, bbox_face in tqdm(train_loader):

        out0 = net(group_topk.to(args.device), topk_normal.to(args.device))
        out1 = net(group_topk.to(args.device), torch.unsqueeze(topk_fevers[3], dim=0).to(args.device))
        out2 = net(group_topk.to(args.device), torch.unsqueeze(topk_fevers[5], dim=0).to(args.device))
        out3 = net(group_topk.to(args.device), torch.unsqueeze(topk_fevers[7], dim=0).to(args.device))
        out4 = net(group_topk.to(args.device), torch.unsqueeze(topk_fevers[9], dim=0).to(args.device))

        loss0 = lossfunc(out0, label0.to(args.device))
        loss1 = lossfunc(out1, label1.to(args.device))
        loss2 = lossfunc(out2, label1.to(args.device))
        loss3 = lossfunc(out3, label1.to(args.device))
        loss4 = lossfunc(out4, label1.to(args.device))

        loss = loss0*(args.alpha) +loss1*(args.beta) +loss2*(args.gamma) +loss3 +loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss0_show += loss0.item()
        loss1_show += loss1.item()
        loss2_show += loss2.item()
        loss_show  += loss.item()
        # pdb.set_trace()

    print('train',
        'loss0-{}-'.format(format(loss0_show/len(train_loader), '.6f')), 
        'loss1-{}-'.format(format(loss1_show/len(train_loader), '.6f')), 
        'loss2-{}-'.format(format(loss2_show/len(train_loader), '.6f')), 
        'loss-{}-'.format(format(loss_show/len(train_loader), '.6f'))
        )

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

    for group_face, group_topk, face_normal, face_fevers, topk_normal, topk_fevers, label0, label1, sample_id_face, bbox_face in tqdm(test_loader):
        
        result = []
        result.append(test_one(args, net, group_topk, topk_normal))
        for topk_fever in topk_fevers:
            topk_fever = torch.unsqueeze(topk_fever, dim=0)
            result.append(test_one(args, net, group_topk, topk_fever))
        results.append([sample_id_face[0], bbox_face[0].numpy(), result])

    return  results


def main():

    if not os.path.exists(args.weights_dir): os.mkdir(args.weights_dir)
    if not os.path.exists(args.results_dir): os.mkdir(args.results_dir)
    cache_dir = os.path.join(args.data_root, 'cache')
    if not os.path.exists(cache_dir): os.mkdir(cache_dir)
    
    # 训练数据队列
    train_set = dataset.Dataset_train(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.bs, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False)

    # 测试数据队列
    test_set = dataset.Dataset_test(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # 模型
    net = model.DGDC(args=args).to(args.device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # 损失函数
    lossfunc = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        print('\n===> epoch: {}/{}'.format(epoch, args.epochs))

        # 训练
        train(train_loader, net, optimizer, lossfunc, args)
        
        # 验证
        # test(val_loader, net, args, epoch)

        # 测试
        results = test(test_loader, net, args)
        
        # 评估
        print('evaluate......')
        utils.evaluate_results(args, results)

        # 保存模型
        weight_file = os.path.join(args.weights_dir, 'train_split{}-train_s{}-train_e{}-epoch{}.pt'.\
                format(args.train_split, args.train_s, args.train_e, epoch))
        print('Save weight to: {}'.format(weight_file))
        torch.save(net.state_dict(), weight_file)


if __name__ == '__main__':

    print('time: ', time.asctime())
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    args = args.get_args()

    main()

