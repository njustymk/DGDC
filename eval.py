import os
import pdb
import args
import numpy as np
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt

def main():
    fever_ids = [3, 5, 7, 9]
    fevers = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    for index, fever in enumerate(fevers):
        print('fever', fever)
        y_label = []
        y_score = []
        results_file = os.path.join(args.results_dir, 'results.txt')
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                # sample_id_face = line[0]
                # xmin = line[1]
                # ymin = line[2]
                # xmax = line[3]
                # ymax = line[4]

                y_label.append(1)
                y_score.append(float(line[5]))

                y_label.append(0)
                y_score.append(float(line[6+index]))

        fpr, tpr, thersholds = roc_curve(y_label, y_score, pos_label=1)
        print(len(fpr))
        # print('----------------------')
        # # print('假阳率\t真阳率\t阈值')
        # for i, value in enumerate(thersholds):
        #     print("%f %f %f" % (fpr[i], tpr[i], value))
        # print('----------------------')

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.3f})'.format(roc_auc), lw=2)
        plt.xlim([-0.05, 1.05])  
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

    plt.savefig(os.path.join(args.results_dir,'roc_dgdc.jpg'))


if __name__ == '__main__':
    args = args.get_args()
    main()