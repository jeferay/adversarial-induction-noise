python3 -u main.py    --version                 resnet18                       \
                      --exp_name                /data/yfwang/Unlearnable/logs/cifar100/minimin \
                      --config_path             configs/cifar100                \
                      --train_data_type         PoisonCIFAR100                  \
                      --test_data_type          CIFAR100                         \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath /data/yfwang/Unlearnable/logs/cifar100/minimin/perturbation.pt \
                      --train
