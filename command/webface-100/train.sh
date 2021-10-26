
EXP_TYPE=mada
DATA_TYPE=webface-10
LABEL_ASSIGN=plus_one
STEP_SIZE=2
TRAIN_STEP=30
DEVICE=6


python3 -u main.py    --version                 resnet18_madrys           \
                      --exp_name                ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_pretrain_epsilon32/main \
                      --device                   ${DEVICE}                      \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         PoisonWebFace-10                 \
                      --test_data_type          WebFace-10                       \
                      --poison_rate             1.0                           \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/samplewise/${DATA_TYPE}/at_pretrain_epsilon32/perturbation/resnet18_madrys/perturbation.pt \
                      --train    &




# clean data

DATA_TYPE=webface-10
DEVICE=1
python3 -u main.py    --version                 resnet18            \
                      --device                  ${DEVICE}                 \
                      --exp_name                ../experiments/clean/${DATA_TYPE}/main \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         WebFace-10                 \
                      --test_data_type          WebFace-10                 \
                      --train_batch_size 48 \
                      --eval_batch_size 48 \
                      --train --load_model  &



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