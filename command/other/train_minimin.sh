python3 -u main.py    --version                 resnet18                       \
                      --exp_name                /data/yfwang/Unlearnable/logs/minimin/rate0.6 \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                  \
                      --poison_rate             0.2                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath /data/yfwang/Unlearnable/logs/minimin/perturbation.pt \
                      --train
