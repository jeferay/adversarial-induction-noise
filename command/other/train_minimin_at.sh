python3 -u main.py    --version                 resnet18_madrys                       \
                      --exp_name                /data/yfwang/Unlearnable/logs/minimin/at \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                  \
                      --poison_rate             1                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath /data/yfwang/Unlearnable/logs/minimin/perturbation.pt \
                      --train
