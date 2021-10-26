python3 -u main.py    --version                 resnet18                       \
                      --exp_name                /data/yfwang/Unlearnable/logs/svhn/minimin \
                      --config_path             configs/svhn                \
                      --train_data_type         PoisonSVHN                  \
                      --test_data_type          SVHN                         \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath /data/yfwang/Unlearnable/logs/svhn/minimin/perturbation.pt \
                      --train
