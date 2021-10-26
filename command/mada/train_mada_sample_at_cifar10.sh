
EXP_TYPE=mada
DATA_TYPE=cifar10
LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=6


python3 -u main.py    --version                 resnet18_madrys            \
                      --exp_name                ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/std_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/main \
                      --device                   ${DEVICE}                      \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         CIFAR10                 \
                      --test_data_type          CIFAR10                       \
                      --poison_rate             1.0                           \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/std_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation/resnet18/perturbation.pt \
                      --train --epochs 120  &




# clean data

DATA_TYPE=cifar10
DEVICE=3
python3 -u main.py    --version                 resnet18_madrys               \
                      --device                  ${DEVICE}                      \
                      --exp_name                ../experiments/clean/${DATA_TYPE}/main \
                      --config_path             ../configs/cifar10                \
                      --train_data_type         CIFAR10                 \
                      --test_data_type          CIFAR10                 \
                      --train   &



EXP_TYPE=random
DATA_TYPE=cifar10

python3 -u main.py    --version                 resnet18_madrys_epsilon32                    \
                      --exp_name                experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/random_epsilon_32/main \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                 \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath experiments/random/samplewise/${DATA_TYPE}/random_epsilon_32/perturbation/resnet18/perturbation.pt \
                      --train --eval --epochs 120 &


/home/wzr/TA-Unlearnable/experiments/mada/samplewise/cifar10/random_epsilon_32/perturbation/resnet18/perturbation.pt