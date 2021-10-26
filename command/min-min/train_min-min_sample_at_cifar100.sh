
EXP_TYPE=min-min
DATA_TYPE=cifar100
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=0


python3 -u main.py    --version                 resnet18_madrys               \
                      --device                  ${DEVICE}                    \
                      --exp_name                ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/std_epsilon32_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/main \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         PoisonCIFAR100                 \
                      --test_data_type          CIFAR100                      \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/std_epsilon32_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation/resnet18/perturbation.pt \
                      --train --epochs 120   &




# clean data

DATA_TYPE=cifar10

python3 -u main.py    --version                 resnet18_madrys               \
                      --exp_name                ../experiments/clean/${DATA_TYPE}/main \
                      --config_path             ../configs/cifar10                \
                      --train_data_type         CIFAR10                 \
                      --test_data_type          CIFAR10                 \
                      --epochs 120 --load_model --res_save_type LL   &



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