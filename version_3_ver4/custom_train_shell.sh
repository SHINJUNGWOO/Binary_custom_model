#!/bin/bash

SAVEDIR='./propose_ptr/version_1'
TRAINNAME="custom_train_2stage.py"
Epoch=360

: <<"END"
python -u ${TRAINNAME} \
    --weight_decay 1e-5 \
    --epochs ${Epoch} \
    --save ${SAVEDIR}/pre

END

python -u ${TRAINNAME} \
    --weight_decay 0 \
    --learning_rate 3e-6 \
    --epochs ${Epoch} \
    --save ${SAVEDIR}/post\
    --binary_w \
#    --pretrained ${SAVEDIR}/pre/model_best.pth.tar \


