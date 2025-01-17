python3 perturbation.py --config_path             configs/svhn                \
                        --exp_name                /data/yfwang/Unlearnable/logs/svhn/minimin \
                        --version                 resnet18                       \
                        --train_data_type         SVHN                       \
                        --test_data_type          SVHN                       \
                        --noise_shape             73257 3 32 32                  \
                        --epsilon                 8                              \
                        --num_steps               20                             \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01                            \
                        --random_start
