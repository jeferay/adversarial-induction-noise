import torch
import os
import numpy as np
import random

def similar_swap(matrix=None, dataset_name = 'CIFAR10'):
    if dataset_name in ['CIFAR100','SVHN','CIFAR10']:
        confusion_matrix_dir = '../experiments/confusion_martix'
        matrix = torch.load(os.path.join(confusion_matrix_dir,dataset_name+'.pth'))
        print("load %s confusion matrix"%dataset_name)


    num_classes = matrix.shape[0]
    left_class = set(range(0,num_classes))
    
    target_label_function = torch.arange(1, num_classes+1, dtype=torch.long)
    target_label_function[-1] = 0

    while(len(left_class)>0):
        left_pair = [(i,j) for i in left_class for j in left_class if j <i]
        pair_score={}
        pair_score = {(i,j):float(matrix[i][j]+matrix[j][i]) for (i,j) in left_pair}
        #print(pair_score)
        selected_pair =sorted(pair_score.items(),key= lambda x:x[1],reverse=True)
        
        (i,j),score =selected_pair[0]
        target_label_function[i]=j
        target_label_function[j]=i
        left_class.remove(i)
        left_class.remove(j)
    
    #print(target_label_function)
    return target_label_function


def least_similar_swap(matrix=None, dataset_name = 'CIFAR10'):
    if dataset_name in ['CIFAR100','SVHN','CIFAR10']:
        confusion_matrix_dir = '../experiments/confusion_martix'
        matrix = torch.load(os.path.join(confusion_matrix_dir,dataset_name+'.pth'))
        print("load %s confusion matrix"%dataset_name)


    num_classes = matrix.shape[0]
    left_class = set(range(0,num_classes))
    
    target_label_function = torch.arange(1, num_classes+1, dtype=torch.long)
    target_label_function[-1] = 0

    while(len(left_class)>0):
        left_pair = [(i,j) for i in left_class for j in left_class if j <i]
        pair_score={}
        pair_score = {(i,j):float(matrix[i][j]+matrix[j][i]) for (i,j) in left_pair}
        #print(pair_score)
        selected_pair =sorted(pair_score.items(),key= lambda x:x[1],reverse=False)
        
        (i,j),score =selected_pair[0]
        target_label_function[i]=j
        target_label_function[j]=i
        left_class.remove(i)
        left_class.remove(j)
    
    print(target_label_function)
    return target_label_function


def random_swap(dataset_name = 'CIFAR10'):
    random.seed(0)
    num_classes = 10
    if dataset_name=='CIFAR100':num_classes=100

    target_label_fucntions=torch.arange(0,num_classes,dtype=torch.long)

    left_label = set(range(int(num_classes/2),num_classes))

    for i in range(int(num_classes/2)):
        select_label = random.sample(list(left_label),1)[0]
        left_label.remove(select_label)
        target_label_fucntions[i]=select_label
        target_label_fucntions[select_label] = i
    print(target_label_fucntions)
    return target_label_fucntions



#return 一个N*1的target矩阵
def random_assign_samplewise(dataset_name='cifar10',num_samples=50000,num_classes=10,seed=0,output=False):
    np.random.seed(0)
    if output:
        label_assign = np.random.randint(0,num_classes,size=[num_samples])
        label_assign = torch.LongTensor(label_assign)
        torch.save(label_assign,f='../experiments/confusion_martix/random.pth')
    else:
        label_assign = torch.load(f='../experiments/confusion_martix/random.pth')
    
    print(label_assign[:10])
    return label_assign


#random_assign_samplewise(output=True)
#random_assign_samplewise(output=False)
#random_swap()