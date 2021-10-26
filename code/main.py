import argparse
import datetime
import os
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import random
import util
import madrys
import mart
import trades
import models

from evaluator import Evaluator
from trainer import Trainer
from label_assign import similar_swap,least_similar_swap
mlconfig.register(madrys.MadrysLoss)
mlconfig.register(mart.MartLoss)
mlconfig.register(trades.TradesLoss)

# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")#version决定了使用的网络以及训练的模式
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)#可以使用多机互连
parser.add_argument('--train', action='store_true', default=False)#决定是否train
parser.add_argument('--eval',action='store_true',default=False)#决定是否单独eval
parser.add_argument('--eval_bias',action='store_true',default=False)#决定eval的时候是否测试的是bias的accu
parser.add_argument('--save_frequency', default=-1, type=int)
parser.add_argument('--epochs', default=0, type=int, help='train epochs')

# Datasets Options
parser.add_argument('--train_face', action='store_true', default=False)
parser.add_argument('--train_portion', default=1.0, type=float)
parser.add_argument('--train_batch_size', default=128, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_path', type=str, default='../datasets')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--poison_rate', default=1.0, type=float)#表示poisoned的image的数量/总image数量
parser.add_argument('--perturb_tensor_filepath', default=None, type=str)
parser.add_argument('--mask_rate',default=1.,type=float)#表示张图中noise被mask的概率，mask为1表示加noise，为0表示像素点不变
parser.add_argument('--output_confusion',action='store_true',default=False)
parser.add_argument('--attack_type',type=str,default='pgd20')
parser.add_argument('--res_save_type',type=str,default=None,choices=['random_assign','LL','MC',None])
parser.add_argument('--device',type=int,default=0)

args = parser.parse_args()


# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")




#set up seed
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
setup_seed(args.seed)

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:%s'%str(args.device))#用于保存数据的特定gpu
    gpus = [args.device]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'#config是由config_path和version共同决定的
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))





def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    train_epochs = config.epochs if args.epochs ==0 else args.epochs
    for epoch in range(starting_epoch, train_epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            ENV['curren_acc'] = evaluator.acc_meters.avg*100
            ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
            
            # Reset Stats #每个epoch结束会reset
            trainer._reset_stats()
            evaluator._reset_stats()
        else:
            pass
            # model.eval()
            # model.module.classify = True
            # evaluator.eval(epoch, model)
            # payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            # logger.info(payload)
            # model.classify = False
            # identity_list = lfw_test.get_lfw_list('lfw_test_pair.txt')
            # img_paths = [os.path.join('../datasets/lfw-112x112', each) for each in identity_list]
            # eval_acc = lfw_test.lfw_test(model, img_paths, identity_list, 'lfw_test_pair.txt', args.eval_batch_size, logger=logger)
            # ENV['curren_acc'] = eval_acc
            # ENV['best_acc'] = max(ENV['best_acc'], eval_acc)
            # ENV['eval_history'].append(eval_acc)
            # # Reset Stats
            # trainer._reset_stats()
            # evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        is_best=is_best,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
        
        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            filename = checkpoint_path_file + '_epoch%d' % (epoch)
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=filename)
            logger.info('Model Saved at %s', filename)

    return

#最后一次train后的模型的eval
def eval(model,evaluator,target_labels = None,output_confusion=False, attack_type='pgd20'):
    # Eval
    confusion_matrix = None
    evaluator._reset_stats()
    logger.info("="*20 + "Eval Epoch %d" % (config.epochs) + "="*20)
    evaluator.eval(config.epochs, model,target_labels = target_labels,attack_type=attack_type)
    payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
    print(payload)
    #logger.info(payload)
    if output_confusion:
        print(evaluator.confusion_matrix)
        confusion_matrix = evaluator.confusion_matrix
        
    evaluator._reset_stats()
    return confusion_matrix

def eval_with_save_res(model,datasets_generator:dataset.DatasetGenerator,res_save_type,attack_type='pgd20'):
    data_loader = datasets_generator.getDataLoader(train_shuffle=False,train_drop_last=False)#不shuffle且保持所有数据
    evaluator = Evaluator(data_loader, logger, config)#建立dataloader不shuffle train data的evaluator
    evaluator._reset_stats()
    logger.info('save the res of %s'%res_save_type)
    res_targets = evaluator.eval_with_save_res(model,res_save_type,attack_type='pgd20')
    torch.save(res_targets,f='../experiments/confusion_martix/'+res_save_type+'.pth')
    logger.info('done')





"""
def eval_all_class(model,evaluator):
    logger.info("="*20 + "Eval Epoch -1" % + "="*20)
    logger.info("eval on all classes")
    for i in range(args.num_classes):
        evaluator._reset_stats()
        evaluator.eval(config.epochs,model,target_labels = torch.LongTensor([i] * args.num_classes).to(device=device))
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        print(payload)
        #logger.info(payload)
        evaluator._reset_stats()
"""



def main():
    model = config.model().to(device)
    datasets_generator = config.dataset(train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        poison_rate=args.poison_rate,#控制poison_rate
                                        mask_rate = args.mask_rate,#控制mask rate
                                        perturb_type=args.perturb_type,
                                        patch_location=args.patch_location,
                                        perturb_tensor_filepath=args.perturb_tensor_filepath,
                                        seed=args.seed)
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))
    if 'Poison' in args.train_data_type and 'Face' not in args.train_data_type:
        with open(os.path.join(exp_path, 'poison_targets.npy'), 'wb') as f:
            if not (isinstance(datasets_generator.datasets['train_dataset'], dataset.MixUp) or isinstance(datasets_generator.datasets['train_dataset'], dataset.CutMix)):
                poison_targets = np.array(datasets_generator.datasets['train_dataset'].poison_samples_idx)
                np.save(f, poison_targets)
                logger.info(poison_targets)
                logger.info('Poisoned: %d/%d' % (len(poison_targets), len(datasets_generator.datasets['train_dataset'])))
                logger.info('Poisoned samples idx saved at %s' % (os.path.join(exp_path, 'poison_targets')))
                logger.info('Poisoned Class %s' % (str(datasets_generator.datasets['train_dataset'].poison_class)))

    if args.train_portion == 1.0:
        data_loader = datasets_generator.getDataLoader()
        train_target = 'train_dataset'
    else:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                               train_shuffle=True,
                                                               train_drop_last=True)

    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()#这里确定了是否是ad training，也就是at的时候,用的是madrys criterion
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)#这里得到的dataloader是之前获取的，对应的train shuffle是true

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': [],
           'cm_history': []}
        
    if args.data_parallel:
        model = torch.nn.DataParallel(model,device_ids = gpus,output_device = gpus[0])

    if args.load_model:
        target_model = model.module if args.data_parallel else model#data_parallel的时候save的是model.module。因此load的时候也要注意
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=target_model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    if args.train:
        train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        
    
    if args.eval:
        logger.info("="*20 + "Eval on original lables" + "="*20)
        confusion_matrix = eval(model,evaluator,output_confusion=args.output_confusion,attack_type=args.attack_type)
        evaluator._reset_stats()#reset stats

        if args.output_confusion:
            confusion_matrix_dir = exp_path
            if not os.path.exists(confusion_matrix_dir):
                os.mkdir(confusion_matrix_dir)
            torch.save(confusion_matrix, f=os.path.join(confusion_matrix_dir,'confusion_matrix'+'.pth'))
            print('confusion matrxi saved at %s'%confusion_matrix_dir)
            print(confusion_matrix)
            
    if args.res_save_type!=None:
        eval_with_save_res(model,datasets_generator,res_save_type=args.res_save_type)

        


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
