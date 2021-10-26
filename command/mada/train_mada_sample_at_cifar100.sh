
#注意对于cifar100 std 要100 epoch
EXP_TYPE=mada
DATA_TYPE=cifar100
LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=6

python3 -u main.py    --version                 resnet18               \
                      --device                   ${DEVICE}                      \
                      --exp_name                ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/main \
                      --config_path             ../configs/cifar100                \
                      --train_data_type         PoisonCIFAR100                 \
                      --test_data_type          CIFAR100                      \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation/resnet18_madrys/perturbation.pt \
                      --train --epochs 100 --load_model   &





# clean data

DATA_TYPE=cifar100
DEVICE=7
python3 -u main.py    --version                 resnet18               \
                      --device                  ${DEVICE}              \
                      --exp_name                ../experiments/clean/${DATA_TYPE}/main \
                      --config_path             ../configs/${DATA_TYPE}               \
                      --train_data_type         CIFAR100                 \
                      --test_data_type         CIFAR100                 \
                      --train --epochs 100 --load_model   &