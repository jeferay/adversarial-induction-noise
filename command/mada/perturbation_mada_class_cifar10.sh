python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                experiments/mada/classwise/cifar10/pretrain_epsilon32_similar/perturbation \
                        --version                 resnet18_madrys                       \
                        --train_data_type         CIFAR10                       \
                        --test_data_type          CIFAR10                        \
                        --noise_shape             50000 3 32 32                  \
                        --epsilon                 32                              \
                        --num_steps               60                             \
                        --step_size               0.8                            \
                        --attack_type             mada                           \
                        --perturb_type            classwise                      \
                        --universal_stop_error    0.01                            \
                        --train_step              30    \
                        --universal_train_target  train_dataset \
                        &

python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                experiments/mada/samplewise/cifar10/at_pretrain_epsilon32_y+1/perturbation \
                        --version                 resnet18_madrys                       \
                        --train_data_type         CIFAR10                       \
                        --test_data_type          CIFAR10                        \
                        --noise_shape             50000 3 32 32                  \
                        --epsilon                 32                              \
                        --num_steps               100                             \
                        --step_size               0.8                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01                            \
                        --train_step              30    \
                        --universal_train_target  train_dataset \
                        --pretrain --load_model \
                        &


