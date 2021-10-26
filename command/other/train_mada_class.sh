
EXP_TYPE=mada
DATA_TYPE=cifar10

python3 -u main.py    --version                 resnet18                    \
                      --exp_name                experiments/${EXP_TYPE}/classwise/${DATA_TYPE}/at_pretrain_epsilon32_similar/main \
                      --config_path             configs/cifar10                \
                      --train_data_type         CIFAR10                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_pretrain_epsilon32_similar/perturbation/resnet18_madrys/perturbation.pt \
                      --train --eval --epochs 60 &
