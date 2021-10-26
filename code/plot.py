
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_res(log_file="experiments/mada/samplewise/cifar10/at_from_scratch_epsilon32_similar/main/resnet18_madrys/resnet18_madrys.log",epoch_num=120):
    nat_accu,rob_accu,loss,nat_accu_on_train,train_loss =[],[],[],[],[]

    with open(log_file,encoding='utf-8') as f:
        res_lines = f.readlines()
        for i,line in enumerate(res_lines):
            line = line.split('\t')
            if len(line)>2 and len(line[-2])>10 and line[-2][:10]=='accpgd_avg':
                nat_accu.append(float(line[3][8:]))
                rob_accu.append(float(line[-2][11:]))
                loss.append(float(line[7][9:]))
            if len(line[0])>21 and line[0][-21:]=='='*20+'\n':
                line = line[0].split(' ')
                if len(line)==5:
                    if line[2]=='====================Eval':
                        train_res = res_lines[i-1].split('\t')
                        if len(train_res)>1:
                            nat_accu_on_train.append(float(train_res[3][8:]))
                            train_loss.append(float(train_res[5][9:]))
                
    nat_accu=nat_accu[:epoch_num]
    rob_accu=rob_accu[:epoch_num]
    loss = loss[:epoch_num]
    nat_accu_on_train=nat_accu_on_train[:epoch_num]
    train_loss=train_loss[:epoch_num]

    return nat_accu,rob_accu,loss,nat_accu_on_train,train_loss
    

    plt.title('Result for natural accu')
    plt.axis([0,120,0,1])
    plt.plot(range(0,120), nat_accu_on_train, color='blue', label='AT on TA poison,train accu')
    plt.plot(range(0,120), nat_accu,color='green',label='AT on TA poison, test accu')
    plt.plot(range(0,120), rob_accu, color='orange',label='AT on TA poison, test robustness')

    
    plt.legend() # 显示图例

    plt.xlabel('epoch')
    plt.ylabel('natural accuracy')
    plt.savefig('at_from_scratch_epsilon32_similar_resnet-madrys_nataccu.jpg')
    


if __name__ == '__main__':
    pass
    

