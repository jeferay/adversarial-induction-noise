# from scratch
LABEL_ASSIGN=plus_one
STEP_SIZE=0.8
TRAIN_STEP=90
DEVICE=6
EPSILON=16
DATA_TYPE=webface-10


python3 perturbation.py --config_path             ../configs/${DATA_TYPE}                \
                        --device                   ${DEVICE}                \
                        --exp_name                ../experiments/mada/samplewise/${DATA_TYPE}/at_from_scratch_epsilon${EPSILON}_${LABEL_ASSIGN}_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation \
                        --version                 resnet18                    \
                        --train_data_type         WebFace-10                       \
                        --noise_shape             5338 3 224 224                  \
                        --epsilon                 ${EPSILON}                             \
                        --num_steps               60                           \
                        --step_size               ${STEP_SIZE}                            \
                        --train_step              ${TRAIN_STEP}                          \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.1                           \
                        --train_step              ${TRAIN_STEP}                    \
                        --random_start                                         \
                        --train_batch_size 48 \
                        --eval_batch_size 48 \
                        --label_assign ${LABEL_ASSIGN} \
                        &


LABEL_ASSIGN=plus_one
STEP_SIZE=2
DEVICE=6
EPSILON=32
DATA_TYPE=webface-10

python3 perturbation.py --config_path             ../configs/${DATA_TYPE}                \
                        --exp_name                ../experiments/mada/samplewise/${DATA_TYPE}/at_pretrain_epsilon${EPSILON}/perturbation \
                        --version                 resnet18_madrys                       \
                        --train_data_type         WebFace-10                      \
                        --test_data_type          WebFace-10                       \
                        --noise_shape             5338 3 224 224                  \
                        --epsilon                 ${EPSILON}                             \
                        --num_steps               200                           \
                        --step_size               0.8                            \
                        --attack_type             mada                           \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.1                            \
                        --train_step              30                              \
                        --label_assign ${LABEL_ASSIGN}       \
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

