

DATA_TYPE=svhn
EXP_TYPE=tom
DEVICE=7

python3 -u main.py    --version                 resnet18_madrys               \
                      --device                  ${DEVICE}                    \
                      --exp_name                ../experiments/${EXP_TYPE}/${DATA_TYPE}/main \
                      --config_path             ../configs/${DATA_TYPE}                \
                      --train_data_type         PoisonSVHN                 \
                      --test_data_type          SVHN                \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise                      \
                      --perturb_tensor_filepath ../experiments/${EXP_TYPE}/${DATA_TYPE}/perturbation.pt \
                      --train  --epochs 120 --load_model &




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
                      --train --eval --epochs 120 --load_model &


/home/wzr/TA-Unlearnable/experiments/mada/samplewise/cifar10/random_epsilon_32/perturbation/resnet18/perturbation.pt