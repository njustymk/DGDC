import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # 简短的测试
    # parser.add_argument('-train_s', type=int, default='0', help='训练数据分段开始处')
    # parser.add_argument('-train_e', type=int, default='1000', help='训练数据分段结束处')
    # parser.add_argument('-test_s', type=int, default='1000', help='测试数据分段开始处')  
    # parser.add_argument('-test_e', type=int, default='2000', help='测试数据分段结束处') 

    parser.add_argument('-train_s', type=int, default='0', help='训练数据分段开始处')
    parser.add_argument('-train_e', type=int, default='3000', help='训练数据分段结束处')
    parser.add_argument('-test_s', type=int, default='3000', help='测试数据分段开始处')  
    parser.add_argument('-test_e', type=int, default='-1', help='测试数据分段结束处') 

    # 数据相关参数
    # parser.add_argument('-train_split', type=str, default='scene2', help='使用数据的日期')
    # parser.add_argument('-test_split', type=str, default='scene2', help='使用数据的日期')  
    # parser.add_argument('-test_epoch', type=int, default=7, help='测试回合')
    # parser.add_argument('-results_dir', type=str, default='results/summer', help='测试结果保存路径')

    parser.add_argument('-train_split', type=str, default='scene8', help='使用数据的日期')
    parser.add_argument('-test_split', type=str, default='scene8', help='使用数据的日期')  
    parser.add_argument('-test_epoch', type=int, default=20, help='测试回合')   #13
    parser.add_argument('-results_dir', type=str, default='results/winter', help='测试结果保存路径')

    # parser.add_argument('-train_split', type=str, default='scene2', help='使用数据的日期')
    # parser.add_argument('-test_split', type=str, default='scene8', help='使用数据的日期')  
    # parser.add_argument('-test_epoch', type=int, default=9, help='测试回合')
    # parser.add_argument('-results_dir', type=str, default='results/summer-winter', help='测试结果保存路径')

    # parser.add_argument('-train_split', type=str, default='scene8', help='使用数据的日期')
    # parser.add_argument('-test_split', type=str, default='scene2', help='使用数据的日期')  
    # parser.add_argument('-test_epoch', type=int, default=13, help='测试回合')
    # parser.add_argument('-results_dir', type=str, default='results/winter-summer', help='测试结果保存路径')

    # *************************************************************************************** #
    parser.add_argument('-temp_offset', type=float, default=0.2, help='温度偏移设定最小值') 
    parser.add_argument('-data_root', type=str, default='../data/data_FFIR', help='数据路径')
    parser.add_argument('-len_sampling', type=int, default=10, help='测试结果保存路径')

    # 面部形状参数设定
    parser.add_argument('-vector_start', type=int, default=0, help='面部向量取值起始位置')
    parser.add_argument('-temp_min', type=int, default=20, help='温度归一化最小值')
    parser.add_argument('-temp_max', type=int, default=40, help='温度归一化最大值')
    parser.add_argument('-fever_gap', type=float, default=0.2, help='发热步长')
    parser.add_argument('-fever_num', type=int, default=10, help='发热数量')
    parser.add_argument('-fever1_train', type=float, default=1.5, help='用于训练的发热温度1')
    parser.add_argument('-fever2_train', type=float, default=2.0, help='用于训练的发热温度2')
    parser.add_argument('-face_h', type=int, default=32, help='面部设定高度')
    parser.add_argument('-face_w', type=int, default=32, help='面部设定宽度')
    parser.add_argument('-vector_s', type=int, default=0, help='面部向量起始位置')
    parser.add_argument('-vector_len', type=int, default=1024, help='面部向量结束位置')

    # 模型相关参数
    parser.add_argument('-end_old', type=int, default=30, help='参与对比的人数')
    parser.add_argument('-len_old', type=int, default=100, help='参与对比的人数')
    parser.add_argument('-num_classes', type=int, default=2, help='类别数量')
    parser.add_argument('-weights_dir', type=str, default='weights/', help='网络参数保存路径')
    parser.add_argument('-temp_step', type=int, default=15, help='面部温度修正参数')

    # 训练相关参数
    parser.add_argument('-device', type=str, default='cuda', help='是否使用显卡')
    parser.add_argument('-lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('-bs', type=int, default=1, help='每次迭代时的批次大小')
    parser.add_argument('-epochs', type=int, default=20, help='训练回合的数量')
    parser.add_argument('-alpha', type=int, default=1, help='loss0所占权重')
    parser.add_argument('-beta', type=int, default=1, help='loss1所占权重')
    parser.add_argument('-gamma', type=int, default=1, help='loss2所占权重')
    parser.add_argument('-delta', type=int, default=1, help='loss3所占权重')
    

    args = parser.parse_args()
    print(args)
    
    return args