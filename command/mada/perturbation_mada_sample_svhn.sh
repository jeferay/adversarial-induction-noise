# from scratch
LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=1
python3 perturbation.py --config_path             ../configs/svhn                \
                        --device                   ${DEVICE}                \
                        --exp_name                ../experiments/mada/samplewise/svhn/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation \
                        --version                 resnet18_madrys                       \
                        --train_data_type         SVHN                      \
                        --test_data_type          SVHN                        \
                        --noise_shape             73257 3 32 32                  \
                        --epsilon                 32                             \
                        --num_steps               60                           \
                        --step_size               ${STEP_SIZE}                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01                            \
                        --train_step              ${TRAIN_STEP}                        \
                        --random_start                                            \
                        --label_assign ${LABEL_ASSIGN} \
                        --iteration_step 3 \
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

