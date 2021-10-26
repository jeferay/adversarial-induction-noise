

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def _get_res(log_file="collection_log/ain_similar32_resnet18_madrys.log",epoch_num=120):
    nat_accu,rob_accu,loss,pgd10_accu_on_train,train_loss =[],[],[],[],[]

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
                            pgd10_accu_on_train.append(float(train_res[3][8:]))
                            train_loss.append(float(train_res[5][9:]))
                
    nat_accu=nat_accu[:epoch_num]
    rob_accu=rob_accu[:epoch_num]
    loss = loss[:epoch_num]
    pgd10_accu_on_train=pgd10_accu_on_train[:epoch_num]
    train_loss=train_loss[:epoch_num]

    return nat_accu,rob_accu,loss,pgd10_accu_on_train,train_loss
    #print('-'*5,len(nat_accu),'-'*5,nat_accu)
    #print('-'*5,len(rob_accu),'-'*5,rob_accu)
    #print('-'*5,len(loss),'-'*5,loss)
    #print('-'*5,len(pgd10_accu_on_train),'-'*5,pgd10_accu_on_train)
    #print('-'*5,len(train_loss),'-'*5,train_loss)

    return 

    plt.title('Result for natural accu')
    plt.axis([0,120,0,1])
    plt.plot(range(0,120), pgd10_accu_on_train, color='blue', label='AT on TA poison,train accu')
    plt.plot(range(0,120), nat_accu,color='green',label='AT on TA poison, test accu')
    plt.plot(range(0,120), rob_accu, color='orange',label='AT on TA poison, test robustness')

    
    plt.legend() # 显示图例

    plt.xlabel('epoch')
    plt.ylabel('natural accuracy')
    plt.savefig('at_from_scratch_epsilon32_similar_resnet-madrys_nataccu.jpg')
    

def get_st_res(log_file='minmin8_resnet18.log',num_epochs=60):
    nat_accu,rob_accu,loss,accu_on_train,train_loss =[],[],[],[],[]

    with open(log_file,encoding='utf-8',mode='r') as f:
        res_lines = f.readlines()
        
        test_res=[]
        train_res=[]
        for i,line in enumerate(res_lines):
            line = line.split(' ')
            #print(line)
            if len(line)>2:
                if line[2]=='====================Eval':
                    test_res.append(res_lines[i+1].split('\t'))
                    train_res.append(res_lines[i-1].split('\t'))
        train_res=train_res[:num_epochs]
        test_res=test_res[:num_epochs]
        

        for i in range(num_epochs):
            nat_accu.append(float(test_res[i][3][8:]))
            rob_accu.append(float(test_res[i][-2][10:]))
            loss.append(float(test_res[i][7][9:]))
            accu_on_train.append(float(train_res[i][3][8:]))
            train_loss.append(float(train_res[i][5][9:]))


    res={'nat_accu':nat_accu,'rob_accu':rob_accu,'loss':loss,'accu_on_train':accu_on_train,'train_loss':train_loss}
    return res


if __name__ == '__main__':
    epoch_num = 100
    #random_res=_get_res(log_file='random_32_resnet18_madrys.log',epoch_num=epoch_num)
    #ain_res = _get_res(log_file="ain_similar32_resnet18_madrys.log",epoch_num=epoch_num)
    #stdpre_res = _get_res(log_file="std_pretrain_32_y+1_resnet18_madrys.log",epoch_num=epoch_num)

    minmin8=get_st_res('minmin8_resnet18_madrys.log',num_epochs=100)
    """
    plt.subplot(231)
    plt.axis([0,epoch_num,0,1])
    plt.plot(range(0,epoch_num), random_res[0], color='blue', label='random32')
    plt.plot(range(0,epoch_num), ain_res[0],color='green',label='asian32')
    plt.plot(range(0,epoch_num), stdpre_res[0], color='orange',label='stdpretrain32')
    plt.title("nat_accu")
    plt.legend() # 显示图例

    plt.subplot(232)
    plt.axis([0,epoch_num,0,1])
    plt.plot(range(0,epoch_num), random_res[1], color='blue', label='random32')
    plt.plot(range(0,epoch_num), ain_res[1],color='green',label='asian32')
    plt.plot(range(0,epoch_num), stdpre_res[1], color='orange',label='stdpretrain32')
    plt.title("rob_accu")
    plt.legend() # 显示图例

    plt.subplot(233)
    #plt.axis([0,epoch_num,0,1])
    plt.plot(range(0,epoch_num), random_res[2], color='blue', label='random32')
    plt.plot(range(0,epoch_num), ain_res[2],color='green',label='asian32')
    plt.plot(range(0,epoch_num), stdpre_res[2], color='orange',label='stdpretrain32')
    
    plt.title("test_loss")
    plt.legend() # 显示图例

    plt.subplot(234)
    plt.axis([0,epoch_num,0,1])
    plt.plot(range(0,epoch_num), random_res[3], color='blue', label='random32')
    plt.plot(range(0,epoch_num), ain_res[3],color='green',label='asian32')
    plt.plot(range(0,epoch_num), stdpre_res[3], color='orange',label='stdpretrain32')
    plt.title("train_accu")
    plt.legend() # 显示图例

    plt.subplot(235)
    #plt.axis([0,epoch_num,0,1])
    plt.plot(range(0,epoch_num), random_res[4], color='blue', label='random32')
    plt.plot(range(0,epoch_num), ain_res[4],color='green',label='asian32')
    plt.plot(range(0,epoch_num), stdpre_res[4], color='orange',label='stdpretrain32')
    plt.title("train_loss")
    plt.legend() # 显示图例
    plt.savefig('sample.png')
    """



    

