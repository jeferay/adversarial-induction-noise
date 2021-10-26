STEP_SIZE=2
TRAIN_STEP=30
DEVICE=3
DATA_TYPE=svhn

python3 perturbation.py --config_path             ../configs/${DATA_TYPE}                \
                        --device                  ${DEVICE}                     \
                        --exp_name                ../experiments/min-min/samplewise/${DATA_TYPE}/std_epsilon32_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation \
                        --version                 resnet18                     \
                        --train_data_type         SVHN                       \
                        --test_data_type          SVHN                       \
                        --noise_shape             73257 3 32 32                  \
                        --epsilon                 32                             \
                        --num_steps               60                           \
                        --step_size               ${STEP_SIZE}                            \
                        --attack_type             min-min                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01                            \
                        --train_step             ${TRAIN_STEP}                        \
                        --random_start                          \
                        &


#pretrain
python3 perturbation.py --config_path             ../configs/cifar10                \
                        --exp_name                ../experiments/mada/samplewise/cifar10/std_pretrain_epsilon8_similar/perturbation \
                        --version                 resnet18                       \
                        --train_data_type         CIFAR10                       \
                        --test_data_type          CIFAR10                       \
                        --noise_shape             50000 3 32 32                  \
                        --epsilon                 8                             \
                        --num_steps               200                           \
                        --step_size               0.8                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.05                            \
                        --train_step              30                              \
                        --pretrain --load_model    &





python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                experiments/mada/classwise/cifar10/at_from_scratch_epsilon16_noiseshape_32_noiseportion_1/perturbation \
                        --version                 resnet18_madrys                       \
                        --train_data_type         CIFAR10                       \
                        --noise_shape             10 3 32 32                  \
                        --epsilon                 16                              \
                        --num_steps               40                             \
                        --step_size               0.8                            \
                        --train_step              20                              \
                        --attack_type             mada                           \
                        --perturb_type            classwise                      \
                        --universal_stop_error    0.03                            \
                        --random_start --universal_train_target  train_dataset --data_parallel &


python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                temp \
                        --version                 resnet18_madrys                       \
                        --train_data_type         CIFAR10                       \
                        --test_data_type          CIFAR10                       \
                        --noise_shape             50000 3 4 4                  \
                        --epsilon                 32                             \
                        --num_steps               60                           \
                        --step_size               0.8                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.1                            \
                        --train_step              30                              \
                        --random_start             &

