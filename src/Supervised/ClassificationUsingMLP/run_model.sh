#!/bin/sh

: '
python model.py --trainingtype withoutSampling --logfile withoutSampling.log \
        --X_train CompleteWithoutSampling/X_train.np --y_train CompleteWithoutSampling/y_train.np \
        --X_valid CompleteWithoutSampling/X_valid.np --y_valid CompleteWithoutSampling/y_valid.np \
        --X_test CompleteWithoutSampling/X_test.np --y_test CompleteWithoutSampling/y_test.np
'y

python model.py --trainingtype withSampling --logfile withSampling.log \
        --X_train CompleteWithSampling/X_train.np --y_train CompleteWithSampling/y_train.np \
        --X_valid CompleteWithSampling/X_valid.np --y_valid CompleteWithSampling/y_valid.np \
        --X_test CompleteWithSampling/X_test.np --y_test CompleteWithSampling/y_test.np

# add attack vs benign, multiAttack classification