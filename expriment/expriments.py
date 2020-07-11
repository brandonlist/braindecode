import torch
import logging
import time

import pandas as pd
import numpy as np

from numpy.random import RandomState
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict
from torch_ext.utils import np_to_var

#signal and target
class SignalAndTarget(object):
    """
    Simple data container class.

    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

def apply_to_X_y(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.

    Applies function to list of X arrays and to list of y arrays separately.

    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects

    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X, y)

def split_into_two_sets(dataset, first_set_fraction=None, n_first_set=None):
    """
    Split set into two sets either by fraction of first set or by number
    of trials in first set.

    Parameters
    ----------
    dataset: :class:`.SignalAndTarget`
    first_set_fraction: float, optional
        Fraction of trials in first set.
    n_first_set: int, optional
        Number of trials in first set

    Returns
    -------
    first_set, second_set: :class:`.SignalAndTarget`
        The two splitted sets.
    """
    assert (first_set_fraction is None) != (
        n_first_set is None
    ), "Pass either first_set_fraction or n_first_set"
    if n_first_set is None:
        n_first_set = int(round(len(dataset.X) * first_set_fraction))
    assert n_first_set < len(dataset.X)
    first_set = apply_to_X_y(lambda a: a[:n_first_set], dataset)
    second_set = apply_to_X_y(lambda a: a[n_first_set:], dataset)
    return first_set, second_set

def concatenate_np_array_or_add_lists(a, b):
    if hasattr(a, "ndim") and hasattr(b, "ndim"):
        new = np.concatenate((a, b), axis=0)
    else:
        if hasattr(a, "ndim"):
            a = a.tolist()
        if hasattr(b, "ndim"):
            b = b.tolist()
        new = a + b
    return new

def concatenate_sets(sets):
    """
    Concatenate all sets together.

    Parameters
    ----------
    sets: list of :class:`.SignalAndTarget`

    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    concatenated_set = sets[0]
    for s in sets[1:]:
        concatenated_set = concatenate_two_sets(concatenated_set, s)
    return concatenated_set

def concatenate_two_sets(set_a, set_b):
    """
    Concatenate two sets together.

    Parameters
    ----------
    set_a, set_b: :class:`.SignalAndTarget`

    Returns
    -------
    concatenated_set: :class:`.SignalAndTarget`
    """
    new_X = concatenate_np_array_or_add_lists(set_a.X, set_b.X)
    new_y = concatenate_np_array_or_add_lists(set_a.y, set_b.y)
    return SignalAndTarget(new_X, new_y)

#logger
log = logging.getLogger(__name__)

class Logger(ABC):
    @abstractmethod
    def log_epoch(self,epoch_dfs):
        raise NotImplementedError('Need to implement the log_epoch function')

class Printer(Logger):
    """
    print output of model training using logging
    print last row of result in a epoch
    """
    def log_epoch(self,epoch_dfs):
        i_epoch = len(epoch_dfs) - 1
        log.info('Epoch {:d}'.format(i_epoch))
        last_row = epoch_dfs.iloc[-1]
        for key,val in last_row.iteritems():
            log.info('{:25s} {:.5f}'.format(key,val))
        log.info('')

class VisdomWriter(Logger):
    pass

#stop_criterion
class MaxEpochs(object):
    """
    Stop when given number of epochs reached:

    Parameters
    ----------
    max_epochs: int
    """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def should_stop(self, epoch_dfs):
        # Keep in mind  epoch 0 without training is also part of dataframe
        return len(epoch_dfs) - 1 >= self.max_epochs

class Or(object):
    """
    Stop when one of the given stop criteria is triggered.

    Parameters
    ----------
    stop_criteria: iterable of stop criteria objects
    """

    def __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria
        self.triggered = dict([(s, False) for s in stop_criteria])

    def should_stop(self, epoch_dfs):
        # Update dictionary of which criterion was triggered ...
        for s in self.stop_criteria:
            self.triggered[s] = s.should_stop(epoch_dfs)
        # Then check if any of them was triggered.
        return np.any(list(self.triggered.values()))

    def was_triggered(self, criterion):
        """
        Return if given criterion was triggered in the last call to should stop.

        Parameters
        ----------
        criterion: stop criterion

        Returns
        -------
        triggered: bool

        """
        return self.triggered[criterion]

class And(object):
    """
    Stop when all of the given stop criteria are triggered.

    Parameters
    ----------
    stop_criteria: iterable of stop criteria objects
    """

    def __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria

    def should_stop(self, epoch_dfs):
        # Update dictionary of which criterion was triggered ...
        for s in self.stop_criteria:
            self.triggered[s] = s.should_stop(epoch_dfs)
        # Then check if all of them were triggered.
        return np.all(list(self.triggered.values()))

    def was_triggered(self, criterion):
        """
        Return if given criterion was triggered in the last call to should stop.

        Parameters
        ----------
        criterion: stop criterion

        Returns
        -------
        triggered: bool

        """
        return self.triggered[criterion]

class NoDecrease(object):
    """ Stops if there is no decrease on a given monitor channel
    for given number of epochs.

    Parameters
    ----------
    metric: str
        Name of metric to monitor for decrease.
    num_epochs: str
        Number of epochs to wait before stopping when there is no decrease.
    min_decrease: float, optional
        Minimum relative decrease that counts as a decrease. E.g. 0.1 means
        only 10% decreases count as a decrease and reset the counter.
    """

    def __init__(self, metric, num_epochs, min_decrease=1e-6):
        self.metric = metric
        self.num_epochs = num_epochs
        self.min_decrease = min_decrease
        self.best_epoch = 0
        self.lowest_val = float("inf")

    def should_stop(self, epoch_dfs):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epoch_dfs) - 1
        current_val = float(epoch_dfs[self.metric].iloc[-1])
        if current_val < ((1 - self.min_decrease) * self.lowest_val):
            self.best_epoch = i_epoch
            self.lowest_val = current_val

        return (i_epoch - self.best_epoch) >= self.num_epochs

class MetricBelow:
    """
    Stops if the given metric is below the given value.

    Parameters
    ----------
    metric: str
        Name of metric to monitor.
    target_value: float
        When metric decreases below this value, criterion will say to stop.
    """

    def __init__(self, metric, target_value):
        self.metric = metric
        self.target_value = target_value

    def should_stop(self, epoch_dfs):
        # -1 due to doing one monitor at start of training
        current_val = float(epoch_dfs[self.metric].iloc[-1])
        return current_val < self.target_value

#monitor
class LossMonitor(object):
    """
    Monitor the examplewise loss.
    """

    def monitor_epoch(self,):
        return

    def monitor_set(
        self,
        setname,
        all_preds,
        all_losses,
        all_batch_sizes,
        all_targets,
        dataset,
    ):
        batch_weights = np.array(all_batch_sizes) / float(
            np.sum(all_batch_sizes)
        )
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        metric = "{:s}_loss".format(setname)
        return {metric: mean_loss}

class MisclassMonitor(object):
    """
    Monitor the examplewise misclassification rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix="misclass", threshold_for_binary_case=None):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            dataset,
    ):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be n_trials * classes * time
            # or just
            # examples * classes
            # make sure not to remove first dimension if it only has size one
            if preds.ndim > 1:
                only_one_row = preds.shape[0] == 1

                pred_labels = np.argmax(preds, axis=1).squeeze()
                # add first dimension again if needed
                if only_one_row:
                    pred_labels = pred_labels[None]
            else:
                assert self.threshold_for_binary_case is not None, (
                    "In case of only one output, please supply the "
                    "threshold_for_binary_case parameter"
                )
                # binary classification case... assume logits
                pred_labels = np.int32(preds > self.threshold_for_binary_case)
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            if targets.ndim > pred_labels.ndim:
                # targets may be one-hot-encoded
                targets = np.argmax(targets, axis=1)
            elif targets.ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(
                    np.expand_dims(targets, extra_dim),
                    pred_labels.shape[extra_dim],
                    extra_dim,
                )
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        metric = "{:s}_{:s}".format(setname, self.col_suffix)
        return {metric: float(misclass)}

class RuntimeMonitor(object):
    """
    Monitor the runtime of each epoch.

    First epoch will have runtime 0.
    """

    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self, ):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        return {"runtime": epoch_runtime}

    def monitor_set(
            self,
            setname,
            all_preds,
            all_losses,
            all_batch_sizes,
            all_targets,
            dataset,
    ):
        return {}


#iterator
def get_balanced_batches(
        n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    """Create indices for batches balanced in size
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches

class BalancedBatchSizeIterator(object):
    """
    Create batches of balanced size.

    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    seed: int
        Random seed for initialization of `numpy.RandomState` random generator
        that shuffles the batches.
    """

    def __init__(self, batch_size, seed=328774):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = RandomState(self.seed)

    def get_batches(self, dataset, shuffle):
        n_trials = len(dataset.X)
        batches = get_balanced_batches(
            n_trials, batch_size=self.batch_size, rng=self.rng, shuffle=shuffle
        )
        for batch_inds in batches:
            batch_X = dataset.X[batch_inds]
            batch_y = dataset.y[batch_inds]

            # add empty fourth dimension if necessary
            if batch_X.ndim == 3:
                batch_X = batch_X[:, None, :, :]
            yield (batch_X, batch_y)

    def reset_rng(self):
        self.rng = RandomState(self.seed)

#recoder
class Model_recorder(object):
    """
    method
    remember_epoch
        record the parameter of the model and optimizer
    reset_to_best_model
        reset parameter of best model
    """
    def __init__(self,metric):
        self.metric = metric
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self,epoch_dfs,model,optimizer):
        """
        Remember parameter values in case this epoch
        has the best performance so far(is the lowest).

        Parameters
        ----------
        epoch_dfs: `pandas.Dataframe`
            Dataframe containing the column `metric` with which performance
            is evaluated.
        model: `torch.nn.Module`
            as to load parameters to model
        optimizer: `torch.optim.Optimizer`
            as to load parameters to optimizer
        """
        i_epoch = len(epoch_dfs) - 1
        current_val = float(epoch_dfs[self.metric].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
        log.info('Best epoch metric {:s} = {:5f} remembered on epoch {:d}'.format(self.metric,current_val,i_epoch))
        log.info('')

    def reset_to_best_model(self, epoch_dfs, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows
        after best epoch from epochs dataframe.

        Modifies parameters of model and optimizer, changes epoch_dfs in-place.

        Parameters
        ----------
        epoch_dfs: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epoch_dfs.drop(range(self.best_epoch + 1, len(epoch_dfs)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)

#fitter
class Model_fitter(object):
    """
    fit a model on datasets(train,valid,test)
    There are ways to train the model:
    1.train on train set for ? epoch
    2.train on train set until stop criterion on valid set
    3.train on train set until stop criterion on valid set,
        reset to the best model and optimizer according to ? criterion
        train on train+valid set until Or(loss on valid set==loss on best train set,
        n_epoch>=2*n_epoch_on_train_set)

    """
    def __init__(self,
                 model,
                 train_set,
                 valid_set,
                 test_set,
                 iterator,
                 loss_function,
                 optimizer,
                 model_constraint,
                 monitors,
                 stop_criterion,
                 do_early_stop=True,
                 run_after_early_stop=True,
                 reset_after_second_run=True,
                 regloss_model=None,
                 cuda=False,
                 pin_memory=False,
                 metric='valid_misclass',
                 loggers=('printer',)
                 ):
        self.__dict__.update(locals())
        del self.self
        if run_after_early_stop or reset_after_second_run:
            assert do_early_stop == True, (
                "Can only run after early stop or "
                "reset after second run if doing an early stop"
            )
        if do_early_stop:
            assert valid_set is not None
        self.datasets = OrderedDict(
            (("train", train_set), ("valid", valid_set), ("test", test_set))
        )
        if valid_set is None:
            self.datasets.pop("valid")
            assert run_after_early_stop == False
            assert do_early_stop == False
        if test_set is None:
            self.datasets.pop("test")
        if self.loggers == ('printer',):
            self.loggers = [Printer()]

    def run(self):
        """
        run trainning
        :return:
        """
        self.setup_trainning()
        log.info('Start Trainning...')
        self.run_until_stop(self.datasets, remember_best=self.do_early_stop)
        if self.do_early_stop:
            # always setup for second stop, in order to get best model
            # even if not running after early stop...
            log.info("Setup for second stop...")
            self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Trainning until second stop...")
            loss_to_reach = float(self.epoch_dfs["train_loss"].iloc[-1])
            datasets = self.datasets
            datasets['train'] = concatenate_sets([datasets['train'],datasets['valid']])
            self.run_until_stop(datasets, remember_best=True)
            if (float(self.epoch_dfs['valid_loss'].iloc[-1])>loss_to_reach) and (self.reset_after_second_run):
                log.info("Resetting to best epoch {:d}".format(
                        self.recorder.best_epoch
                    ))
                self.recorder.reset_to_best_model((self.epoch_dfs,self.model,self.optimizer))
    def setup_trainning(self):
        """
        1.move to cuda
        2.reset recorder
        3.setup epoch_dfs
        :return:
        """
        #move to cuda if requested
        if self.cuda:
            assert torch.cuda.is_available() ,'Cuda not available'
            self.model.cuda()
        #reset if fit again
        if self.do_early_stop:
            self.recorder = Model_recorder(self.metric)
        #setup epoch_dataframe
        self.epoch_dfs = pd.DataFrame()

    def setup_after_stop_training(self):
        """
        Setup training after first stop.

        Resets parameters to best parameters and updates stop criterion.
        """
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.before_stop_df = deepcopy(self.epoch_dfs)
        self.recorder.reset_to_best_model(
            self.epoch_dfs, self.model, self.optimizer
        )
        loss_to_reach = float(self.epoch_dfs["train_loss"].iloc[-1])
        self.stop_criterion = Or(
            stop_criteria=[
                MaxEpochs(max_epochs=self.recorder.best_epoch * 2),
                MetricBelow(
                    metric="valid_loss", target_value=loss_to_reach
                ),
            ]
        )
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))

    def train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.

        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`
        """
        self.model.train()
        if type(inputs) == np.ndarray:
            inputs = np_to_var(inputs, pin_memory=self.pin_memory)
        if type(targets) == np.ndarray:
            targets = np_to_var(targets, pin_memory=self.pin_memory)
        if self.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)
        if self.regloss_model is not None:
            loss = loss + self.regloss_model(self.model)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def run_one_epoch(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets for one epoch.

        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters if this epoch is best epoch.
        """
        batch_generator = self.iterator.get_batches(
            datasets['train'], shuffle=True
        )
        start_train_epoch_time = time.time()
        for inputs, targets in batch_generator:
            if len(inputs) > 0:
                self.train_batch(inputs, targets)
        end_train_epoch_time = time.time()
        log.info(
            "Time spent for training updates: {:.2f}s".format(
                end_train_epoch_time - start_train_epoch_time
            )
        )

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.recorder.remember_epoch(
                self.epoch_dfs, self.model, self.optimizer
            )

    def run_until_stop(self,datasets,remember_best):
        """
        Run trainning and evaluation on given datasets until stop criterion fulfilled
        :param datasets: OrderedDict
        :param remember_best: bool
            whether to remember paras at best epoch
        :return:
        """
        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.recorder.remember_epoch(self.epoch_dfs,self.model,self.optimizer)
        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epoch_dfs):
            self.run_one_epoch(datasets, remember_best)

    def log_epoch(self):
        for logger in self.loggers:
            logger.log_epoch(self.epoch_dfs)

    def eval_on_batch(self,inputs,targets):
        """

        :param inputs: torch.autograd.Variable
        :param targets: torch.autograd.Variable
        :return:
        """
        self.model.eval()
        with torch.no_grad():
            if type(inputs) == np.ndarray:
                inputs = np_to_var(inputs, pin_memory=self.pin_memory)
            if type(targets) == np.ndarray:
                targets = np_to_var(targets, pin_memory=self.pin_memory)
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().detach().numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().detach().numpy() for o in outputs]
            loss = loss.cpu().detach().numpy()
        return outputs, loss

    def monitor_epoch(self,datasets):
        """
        1.evaluate one epoch on given dataset
        2.append epoch_dfs
        :param datasets:
        :return:
        """
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)

        for setname in datasets:
            dataset = datasets[setname]
            batch_generator = self.iterator.get_batches(dataset,shuffle=False)

            #get n_batchs
            if hasattr(batch_generator, "__len__"):
                n_batches = len(batch_generator)
            else:
                n_batches = sum(1 for i in batch_generator)
                batch_generator = self.iterator.get_batches(
                    dataset, shuffle=False
                )
            all_preds,all_targets = None, None
            all_losses,all_batch_sizes = [], []
            for inputs,targets in batch_generator:
                preds, loss = self.eval_on_batch(inputs, targets)
                all_losses.append(loss)
                all_batch_sizes.append(len(targets))
                if all_preds is None:
                    assert all_targets is None
                    if len(preds.shape) == 2:
                        # first batch size is largest
                        max_size, n_classes = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (n_batches * max_size, n_classes), dtype=np.float32
                        )
                    else:
                        assert len(preds.shape) == 3
                        # first batch size is largest
                        max_size, n_classes, n_preds_per_input = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (
                                n_batches * max_size,
                                n_classes,
                                n_preds_per_input,
                            )  ,
                            dtype=np.float32,
                        )
                    all_preds[: len(preds)] = preds
                    all_targets = np.nan * np.ones((n_batches * max_size))
                    all_targets[: len(targets)] = targets
                else:
                    start_i = sum(all_batch_sizes[:-1])
                    stop_i = sum(all_batch_sizes)
                    all_preds[start_i:stop_i] = preds
                    all_targets[start_i:stop_i] = targets
            self.check = all_preds
            # check for unequal batches
            unequal_batches = len(set(all_batch_sizes)) > 1
            all_batch_sizes = sum(all_batch_sizes)
            # remove nan rows in case of unequal batch sizes
            if unequal_batches:
                assert np.sum(np.isnan(all_preds[: all_batch_sizes - 1])) == 0
                assert np.sum(np.isnan(all_preds[all_batch_sizes:])) > 0
                # TODO: is there a reason we dont just take
                # all_preds = all_preds[:all_batch_sizes] and
                # all_targets = all_targets[:all_batch_sizes] ?
                range_to_delete = range(all_batch_sizes, len(all_preds))
                all_preds = np.delete(all_preds, range_to_delete, axis=0)
                all_targets = np.delete(all_targets, range_to_delete, axis=0)
            assert (
                    np.sum(np.isnan(all_preds)) == 0
            ), "There are still nans in predictions"
            assert (
                    np.sum(np.isnan(all_targets)) == 0
            ), "There are still nans in targets"
            # add empty dimension
            # monitors expect n_batches x ...
            all_preds = all_preds[np.newaxis, :]
            all_targets = all_targets[np.newaxis, :]
            all_batch_sizes = [all_batch_sizes]
            all_losses = [all_losses]
            for m in self.monitors:
                result_dict = m.monitor_set(
                    setname,
                    all_preds,
                    all_losses,
                    all_batch_sizes,
                    all_targets,
                    dataset,
                )
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epoch_dfs = self.epoch_dfs.append(row_dict, ignore_index=True)
        assert set(self.epoch_dfs.columns) == set(row_dict.keys()), (
            "Columns of dataframe: {:s}\n and keys of dict {:s} not same"
        ).format(str(set(self.epoch_dfs.columns)), str(set(row_dict.keys())))
        self.epoch_dfs = self.epoch_dfs[list(row_dict.keys())]



import sys
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)
import torch.nn.functional as F
from torch import optim
from BCICdatasets import BCICompetition4_2a
from expriment.experiment_utils import split_dataset
from torch.utils.data import DataLoader
from BCI_database import Subject_dataset
import numpy as np
import torch

db = BCICompetition4_2a()
db.load_data()
train_set = db.create_signal_and_target(0)
test_set = db.create_signal_and_target(0)
train_set, valid_set = split_into_two_sets(
    train_set, first_set_fraction=1 - 0.2
)

cuda = True
batch_size = 60
max_epochs = 1000
max_increase_epochs = 200

stop_criterion = Or(
    [
        MaxEpochs(max_epochs),
        NoDecrease("valid_misclass", max_increase_epochs),
    ]
)


time_steps = train_set.X[0].shape[1]

from models.CNN import ShallowConvNet
model = ShallowConvNet(in_chans=22, time_steps=time_steps, classes=4, fs=250,linear_init_std=0.8)
model.apply(model.weigth_init)
log.info("Model: \n{:s}".format(str(model)))

optimizer = optim.Adam(model.parameters())

iterator = BalancedBatchSizeIterator(batch_size=batch_size)


monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

from torch_ext.utils import MaxNormDefaultConstraint
model_constraint = MaxNormDefaultConstraint()

exp = Model_fitter(
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
    cuda=cuda,
)
exp.run()