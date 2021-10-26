python3 -u main.py    --version                 resnet18                       \
                      --exp_name                /data/yfwang/Unlearnable/logs/minimax \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                  \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath /data/yfwang/Unlearnable/logs/minimax/perturbation.pt \
                      --train
