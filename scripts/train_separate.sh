#!/bin/bash

N_TRAIN=50000
N_TEST=5000
EPOCHS=1
BATCH_SIZE=128

# QM9
python train.py -d QM9 -t energy --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE

# MD17 Energy
python train.py -d MD17 -t energy -m aspirin --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m azobenzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m benzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m ethanol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m malonaldehyde --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m naphthalene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m paracetamol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m salicylic_acid --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m toluene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy -m uracil --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE

# MD17 Force
python train.py -d MD17 -t force -m aspirin --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m azobenzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m benzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m ethanol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m malonaldehyde --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m naphthalene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m paracetamol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m salicylic_acid --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m toluene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t force -m uracil --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE

# MD17 Energy+Force
python train.py -d MD17 -t energy_and_force -m aspirin --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m azobenzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m benzene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m ethanol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m malonaldehyde --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m naphthalene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m paracetamol --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m salicylic_acid --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m toluene --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d MD17 -t energy_and_force -m uracil --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE

# ISO17
python train.py -d ISO17 -t energy --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d ISO17 -t force --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE
python train.py -d ISO17 -t energy_and_force --n_train $N_TRAIN --n_test $N_TEST -e $EPOCHS -lr 0.01 --batch_size $BATCH_SIZE

