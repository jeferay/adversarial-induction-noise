
EXP_TYPE=mada
DATA_TYPE=svhn
LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=6


python3 -u main.py    --version                 resnet18               \
                      --device                   ${DEVICE}                      \
                      --exp_name                ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/main \
                      --config_path             ../configs/svhn                \
                      --train_data_type         PoisonSVHN                 \
                      --test_data_type          SVHN                       \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_from_scratch_epsilon32_${LABEL_ASSIGN}_batch128_rs_stepsize${STEP_SIZE}_trainstep${TRAIN_STEP}/perturbation/resnet18_madrys/perturbation.pt \
                      --train --eval --epochs 60 --load_model  &




# clean data

DATA_TYPE=svhn
DEVICE=6

python3 -u main.py    --version                 resnet18               \
                      --device                  ${DEVICE}              \
                      --exp_name                ../experiments/clean/${DATA_TYPE}/main \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         SVHN                 \
                      --test_data_type         SVHN                 \
                      --train --epochs 60 &