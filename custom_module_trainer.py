"""
ModuleTrainer for high level training on Pytorch models
"""
from __future__ import print_function
from __future__ import absolute_import

import torchsample
import functools
import math
import torch as th
from torch.autograd import Variable

from torchsample.modules._utils import (_is_tuple_or_list, _add_regularizer_to_loss_fn,
                                        _parse_num_inputs_and_targets_from_loader)
from torchsample.callbacks import CallbackContainer, TQDM
from torchsample.regularizers import RegularizerCallback
from torchsample.constraints import ConstraintCallback
from torchsample.metrics import MetricContainer, MetricCallback


class CustomModuleTrainer(torchsample.modules.ModuleTrainer):
    def __init__(self, model):
        super().__init__(model)

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   initial_epoch=0,
                   num_epoch=100,
                   cuda_device=-1,
                   verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        self.model.train(mode=True)
        # ----------------------------------------------------------------------
        num_inputs = loader.dataset.num_inputs
        num_targets = loader.dataset.num_targets
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size

        if val_loader is not None:
            num_val_inputs = val_loader.dataset.num_inputs
            num_val_targets = val_loader.dataset.num_targets
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn,
                                                          self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks + tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': loader.batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            for epoch_idx in range(initial_epoch, num_epoch):
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                batch_logs = {}
                loader_iter = iter(loader)
                for batch_idx in range(num_batches):

                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, target_batch = fit_helper.grab_batch_from_loader(loader_iter)
                    if cuda_device >= 0:
                        input_batch, target_batch = fit_helper.move_to_cuda(cuda_device, input_batch, target_batch)

                    # ---------------------------------------------
                    self._optimizer.zero_grad()
                    output_batch, _ = fit_forward_fn(input_batch)
                    loss = fit_loss_fn(output_batch, target_batch)
                    loss.backward()
                    self._optimizer.step()
                    # ---------------------------------------------

                    if self._has_regularizers:
                        batch_logs['reg_loss'] = self.regularizer_container.current_value
                    if self._has_metrics:
                        metrics_logs = self.metric_container(output_batch, target_batch)
                        batch_logs.update(metrics_logs)

                    # batch_logs['loss'] = loss.data[0]
                    batch_logs['loss'] = loss.item()
                    callback_container.on_batch_end(batch_idx, batch_logs)

                epoch_logs.update(self.history.batch_metrics)
                if has_val_data:
                    val_epoch_logs = self.evaluate_loader(val_loader,
                                                          cuda_device=cuda_device,
                                                          verbose=verbose)
                    self._in_train_loop = False
                    # self.history.batch_metrics.update(val_epoch_logs)
                    # epoch_logs.update(val_epoch_logs)
                    epoch_logs.update(val_epoch_logs)
                    epoch_logs.update(batch_logs)
                    # TODO how to fix this?
                    # self.history.batch_metrics.update(val_epoch_logs)

                callback_container.on_epoch_end(epoch_idx, epoch_logs)

                if self._stop_training:
                    break
        self.model.train(mode=False)

    def evaluate_loader(self,
                        loader,
                        cuda_device=-1,
                        verbose=1):
        self.model.train(mode=False)
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = _get_helper(self, num_inputs, num_targets)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs = {'val_loss': 0.}
        loader_iter = iter(loader)

        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()

        samples_seen = 0
        for batch_idx in range(num_batches):
            input_batch, target_batch = evaluate_helper.grab_batch_from_loader(loader_iter, volatile=True)
            if cuda_device >= 0:
                input_batch, target_batch = evaluate_helper.move_to_cuda(cuda_device, input_batch, target_batch)

            self._optimizer.zero_grad()
            output_batch, _ = eval_forward_fn(input_batch)
            loss = eval_loss_fn(output_batch, target_batch)

            samples_seen += batch_size
            eval_logs['val_loss'] = (samples_seen * eval_logs['val_loss'] + loss.item() * batch_size) / (
                        samples_seen + batch_size)

            if self._has_metrics:
                metrics_logs = metric_container(output_batch, target_batch)
                eval_logs.update(metrics_logs)

        self.model.train(mode=True)
        return eval_logs


def _get_helper(trainer, num_inputs, num_targets):
    if (num_inputs == 1) and (num_targets == 1):
        helper = SingleInput_SingleTarget_Helper()

    elif (num_inputs == 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_tuple_or_list(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        helper = SingleInput_MultiTarget_Helper()

    elif (num_inputs == 1) and (num_targets == 0):
        helper = SingleInput_NoTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 1):
        helper = MultiInput_SingleTarget_Helper()

    elif (num_inputs > 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_tuple_or_list(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        helper = MultiInput_MultiTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 0):
        helper = MultiInput_NoTarget_Helper()

    return helper


class SingleInput_SingleTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = targets.cuda(cuda_device)
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets, volatile=False):
        input_batch = Variable(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
        target_batch = Variable(targets[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile,
                                requires_grad=False)
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch, target_batch = next(loader_iter)
        return Variable(input_batch, volatile=volatile), Variable(target_batch, volatile=volatile, requires_grad=False)

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = tforms[1](target_batch)
        input_batch, target_batch = tforms[2](input_batch, target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
        # def new_loss_fn(output_batch, target_batch):
        #    return self.calculate_loss(output_batch, target_batch, loss_fn)
        # return new_loss_fn


class SingleInput_MultiTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = [target_[rand_indices] for target_ in targets]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets, volatile=False):
        input_batch = Variable(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
        target_batch = [Variable(target_[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile,
                                 requires_grad=False)
                        for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch, target_batch = next(loader_iter)
        return Variable(input_batch, volatile=volatile), [Variable(target_, volatile=volatile, requires_grad=False) for
                                                          target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx])
                    for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_SingleTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        targets = targets.cuda(cuda_device)
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets, volatile=False):
        input_batch = [Variable(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
                       for input_ in inputs]
        target_batch = Variable(targets[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile,
                                requires_grad=False)
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch, target_batch = next(loader_iter)
        return [Variable(input_, volatile=volatile) for input_ in input_batch], Variable(target_batch,
                                                                                         volatile=volatile,
                                                                                         requires_grad=False)

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_MultiTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = [input_[rand_indices] for input_ in inputs]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets, volatile=False):
        input_batch = [Variable(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
                       for input_ in inputs]
        target_batch = [Variable(target_[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile,
                                 requires_grad=False)
                        for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch, target_batch = next(loader_iter)
        return [Variable(input_, volatile=volatile) for input_ in input_batch], [
            Variable(target_, volatile=volatile, requires_grad=False) for target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx])
                    for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class SingleInput_NoTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = inputs.cuda(cuda_device)
        return inputs, None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None, volatile=False):
        input_batch = Variable(inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch = next(loader_iter)
        return Variable(input_batch, volatile=volatile), None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = tforms[0](input_batch)
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_NoTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        return inputs, None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None, volatile=False):
        input_batch = [Variable(input_[batch_idx * batch_size:(batch_idx + 1) * batch_size], volatile=volatile)
                       for input_ in inputs]
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter, volatile=False):
        input_batch = next(loader_iter)
        return [Variable(input_, volatile=volatile) for input_ in input_batch], None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
