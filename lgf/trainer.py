import os
from collections import Counter
import struct

import numpy as np

import torch
import torch.nn.utils

from ignite.engine import Events, Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import RunningAverage, Metric, Loss
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, GradsScalarHandler


class AverageMetric(Metric):
    def reset(self):
        self._sums = Counter()
        self._num_examples = Counter()

    def update(self, output):
        for k, v in output.items():
            self._sums[k] += torch.sum(v)
            self._num_examples[k] += torch.numel(v)

    def compute(self):
        return {k: v / self._num_examples[k] for k, v in self._sums.items()}

    def completed(self, engine):
        engine.state.metrics = {**engine.state.metrics, **self.compute()}

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)


class Trainer:
    _STEPS_PER_LOSS_WRITE = 10
    _STEPS_PER_GRAD_WRITE = 10
    _STEPS_PER_LR_WRITE = 10

    def __init__(
            self,

            module,
            device,

            train_loss,
            train_loader,
            opt,
            lr_scheduler,
            max_epochs,
            max_grad_norm,

            test_metrics,
            test_loader,
            epochs_per_test,

            early_stopping,
            valid_loss,
            valid_loader,
            max_bad_valid_epochs,

            visualizer,

            writer,
            should_checkpoint_latest,
            should_checkpoint_best_valid,

            config
    ):
        self._module = module
        self._module.to(device)
        self._device = device

        self._train_loss = train_loss
        self._train_loader = train_loader
        self._opt = opt
        self._lr_scheduler = lr_scheduler
        self._max_epochs = max_epochs
        self._max_grad_norm = max_grad_norm

        self._test_metrics = test_metrics
        self._test_loader = test_loader
        self._epochs_per_test = epochs_per_test

        self._valid_loss = valid_loss
        self._valid_loader = valid_loader
        self._max_bad_valid_epochs = max_bad_valid_epochs
        self._best_valid_loss = float("inf")
        self._num_bad_valid_epochs = 0

        self._visualizer = visualizer

        self._writer = writer
        self._should_checkpoint_best_valid = should_checkpoint_best_valid

        self.config = config

        ### Training

        self._trainer = Engine(self._train_batch)

        AverageMetric().attach(self._trainer)
        ProgressBar(persist=True).attach(self._trainer, ["loss"])

        self._trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.train())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_info)

        if should_checkpoint_latest:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._save_checkpoint("latest"))

        ### Testing

        self._tester = Engine(self._test_batch)

        AverageMetric().attach(self._tester)
        ProgressBar(persist=False, desc="Testing").attach(self._tester)

        if self.config['test_every_epoch']:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._test_hook)
        self._tester.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.eval())

        ### Validation

        if early_stopping:
            self._validator = Engine(self._validate_batch)

            AverageMetric().attach(self._validator)
            ProgressBar(persist=False, desc="Validating").attach(self._validator)

            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, self._validate)
            self._validator.add_event_handler(Events.EPOCH_STARTED, lambda _: self._module.eval())

    def train(self):
        self._trainer.run(data=self._train_loader, max_epochs=self._max_epochs)

    def _train_batch(self, engine, batch):
        x, _ = batch # TODO: Potentially pass y also for genericity
        x = x.to(self._device)

        self._opt.zero_grad()

        loss = self._train_loss(self._module, x).mean()
        loss.backward()

        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._module.parameters(), self._max_grad_norm)

        self._opt.step()

        if self.config["lr_schedule"] != 'plateau':
            self._lr_scheduler.step()

        return {"loss": loss}

    @torch.no_grad()
    def _test(self, engine):
        epoch = engine.state.epoch

        state = self._tester.run(data=self._test_loader)

        for k, v in state.metrics.items():
            self._writer.write_scalar(f"test/{k}", v, global_step=engine.state.epoch)

        # TODO this is ugly would be great to make this part of the Writer class
        # Open the file in the append-binary mode
        with open(os.path.join(self._writer._logdir, 'test_loss.dat'), 'ab') as f:
            f.write(struct.pack('if', epoch, state.metrics['elbo'].item()))

        self._visualizer.visualize(self._module, epoch)

    def _test_hook(self, engine):
        epoch = engine.state.epoch
        if (epoch - 1) % self._epochs_per_test == 0: # Test after first epoch
            self._test(engine)

    def _test_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return self._test_metrics(self._module, x)

    @torch.no_grad()
    def _validate(self, engine):
        state = self._validator.run(data=self._valid_loader)
        valid_loss = state.metrics["loss"]

        if self.config["lr_schedule"] == 'plateau':
            self._lr_scheduler.step(valid_loss)

        # TODO this is ugly, it would be great to wrap this in the Writer class
        # Open the file in the append-binary mode
        with open(os.path.join(self._writer._logdir, 'validation_loss.dat'), 'ab') as f:
            f.write(struct.pack('if', engine.state.epoch, valid_loss.item()))

        for k, v in state.metrics.items():
            self._writer.write_scalar(f"valid/{k}", v, global_step=engine.state.epoch)

        if valid_loss < self._best_valid_loss:
            print(f"Best validation loss {valid_loss} after epoch {engine.state.epoch}")
            self._num_bad_valid_epochs = 0
            self._best_valid_loss = valid_loss

            if self._should_checkpoint_best_valid:
                self._save_checkpoint(tag="best_valid")

            if not self.config['test_every_epoch'] and \
                    engine.state.epoch > self.config['no_test_until_epoch']:
                self._test(engine)

        else:
            self._num_bad_valid_epochs += 1

            # We do this manually (i.e. don't use Ignite's early stopping) to permit
            # saving/resuming more easily
            if self._num_bad_valid_epochs > self._max_bad_valid_epochs:
                self._test(engine)
                print(
                    f"No validation improvement after {self._num_bad_valid_epochs} epochs. Terminating."
                )
                self._trainer.terminate()

    def _validate_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return {"loss": self._valid_loss(self._module, x)}

    def _log_training_info(self, engine):
        i = engine.state.iteration

        if i % self._STEPS_PER_LOSS_WRITE == 0:
            loss = engine.state.output["loss"]
            self._writer.write_scalar("train/loss", loss, global_step=i)

        # TODO: Inefficient to recompute this if we are doing gradient clipping
        if i % self._STEPS_PER_GRAD_WRITE == 0:
            self._writer.write_scalar("train/grad-norm", self._get_grad_norm(), global_step=i)

        # TODO: We should do this _before_ calling self._lr_scheduler.step(), since
        # we will not correspond to the learning rate used at iteration i otherwise
        if i % self._STEPS_PER_LR_WRITE == 0:
            self._writer.write_scalar("train/lr", self._get_lr(), global_step=i)

    def _get_grad_norm(self):
        norm = 0
        for param in self._module.parameters():
            if param.grad is not None:
                norm += param.grad.norm().item()**2
        return np.sqrt(norm)

    def _get_lr(self):
        param_group, = self._opt.param_groups
        return param_group["lr"]

    def _save_checkpoint(self, tag):
        # We do this manually (i.e. don't use Ignite's checkpointing) because
        # Ignite only allows saving objects, not scalars (e.g. the current epoch) 
        checkpoint = {
            "epoch": self._trainer.state.epoch,
            "iteration": self._trainer.state.iteration,
            "module_state_dict": self._module.state_dict(),
            "opt_state_dict": self._opt.state_dict(),
            "best_valid_loss": self._best_valid_loss,
            "num_bad_valid_epochs": self._num_bad_valid_epochs
        }

        self._writer.write_checkpoint(tag, checkpoint)
