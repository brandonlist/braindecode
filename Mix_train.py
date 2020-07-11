import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

from modules import band_pass_EEGdata

log = logging.getLogger(__name__)

from braindecode.datautil.signalproc import exponential_running_standardize



logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)

from BCICdatasets import BCIC4_2a
from experiment_utils import split_dataset
from torch.utils.data import DataLoader
from BCI_database import Subject_dataset
import numpy as np
import torch

db = BCIC4_2a(uv=True)
db.load_data()
db_0 = Subject_dataset(db,1)
fl = 4
fh = 37
train_db,valid_db,test_db = split_dataset(dataset=db_0,train_rate=0.8,valid_rate=0.1)

train_X,train_Y = next(iter(DataLoader(train_db,len(train_db))))
valid_X,valid_Y = next(iter(DataLoader(valid_db,len(valid_db))))
test_X,test_Y = next(iter(DataLoader(test_db,len(test_db))))

train_X = train_X.to(dtype=torch.float32).numpy()
train_y = (train_Y.numpy()-1).astype(np.int64)
valid_X = valid_X.to(dtype=torch.float32).numpy()
valid_y = (valid_Y.numpy()-1).astype(np.int64)
test_X = test_X.to(dtype=torch.float32).numpy()
test_y = (test_Y.numpy()-1).astype(np.int64)

# train_X = band_pass_EEGdata(eegdata=train_X,lf=fl,hf=fh,fs=db.fs)
# valid_X = band_pass_EEGdata(eegdata=valid_X,lf=fl,hf=fh,fs=db.fs)
# test_X = band_pass_EEGdata(eegdata=test_X,lf=fl,hf=fh,fs=db.fs)


train_X = train_X.astype(np.float32)
valid_X = valid_X.astype(np.float32)
test_X = test_X.astype(np.float32)


in_chan = db.signal_shape()[0]
time_steps = db.signal_shape()[1]
from braindecode.datautil.signal_target import SignalAndTarget
train_set = SignalAndTarget(train_X, y=train_y)
valid_set = SignalAndTarget(valid_X, y=valid_y)
test_set = SignalAndTarget(test_X, y=valid_y)

cuda = True
batch_size = 60
max_epochs = 20000
max_increase_epochs = 360

model = ShallowFBCSPNet(in_chan, db.n_classes, input_time_length=time_steps,
                        final_conv_length="auto").create_network()
log.info("Model: \n{:s}".format(str(model)))

optimizer = optim.Adam(model.parameters())

iterator = BalancedBatchSizeIterator(batch_size=batch_size)

stop_criterion = Or(
    [
        MaxEpochs(max_epochs),
        NoDecrease("valid_misclass", max_increase_epochs),
    ]
)

monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

model_constraint = MaxNormDefaultConstraint()

exp = Experiment(
    model,
    train_set,
    valid_set,
    test_set,
    iterator=iterator,
    loss_function=F.nll_loss,
    optimizer=optimizer,
    model_constraint=model_constraint,
    monitors=monitors,
    stop_criterion=stop_criterion,
    remember_best_column="valid_misclass",
    run_after_early_stop=True,
    cuda=cuda,
)
exp.run()

