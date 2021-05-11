#!/bin/bash

SAVEDIR='./react_ptr/version_2'
TRAINNAME="custom_train_2stage.py"
Epoch=256

#: <<"END"
python -u ${TRAINNAME} \
    --weight_decay 1e-5 \
    --epochs ${Epoch} \
    --save ${SAVEDIR}/pre

#END

python -u ${TRAINNAME} \
    --weight_decay 0 \
    --epochs ${Epoch} \
    --save ${SAVEDIR}/post\
    --binary_w \
    --pretrained ${SAVEDIR}/pre/model_best.pth.tar \


