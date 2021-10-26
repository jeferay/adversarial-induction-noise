# from scratch

LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=7
python3 perturbation.py --config_path             ../configs/cifar100               \
                        --device                   ${DEVICE}                \
                        --exp_name                ../experiments/mada/samplewise/cifar100/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation \
                        --version                 resnet18_madrys                      \
                        --train_data_type         CIFAR100                       \
                        --test_data_type          CIFAR100                       \
                        --noise_shape             50000 3 32 32                  \
                        --epsilon                 32                             \
                        --num_steps               60                           \
                        --step_size               ${STEP_SIZE}                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.01                            \
                        --train_step              ${TRAIN_STEP}                        \
                        --random_start                                   \
                        --label_assign ${LABEL_ASSIGN} \
                        &
