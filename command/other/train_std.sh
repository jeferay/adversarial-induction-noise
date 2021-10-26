python3 -u main.py    --version                 resnet18                       \
                      --exp_name                path/to/your/experiment/folder \
                      --config_path             configs/cifar10                \
                      --train_data_type         CIFAR10                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath path/to/your/experiment/folder/perturbation.pt \
                      --train