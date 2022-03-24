import torch
import numpy as np
import os
from typing import Optional, Tuple, List, Callable
import math
import logging
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
import contextlib
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger('tfs')


class Average:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class SingleDeviceLMTrainer:
    """Simple trainer that works on a single machine/device
    """

    def __init__(
            self,
            model,
            lr: float = 1.0e-4,
            batch_size: int = 256,
            weight_decay: float = 1.0e-2,
            warmup_fract: int = 0.1,
            plateau_fract: int = 0.0,
            decay_type: str = 'cosine',
            total_steps: Optional[int] = None,
            global_step: int = 0,
            alpha_decay: float = 0.0,
            betas: Tuple[float] = (0.9, 0.98),
            eps=1e-08,
            grad_clip: float = 1.0,
            num_train_workers=4,
            num_valid_workers=1,
            dont_decay_weights: Optional[List[str]] = None,
            collate_function: Optional[Callable] = None,
            **kwargs,
    ):

        if weight_decay == 0.0:
            parameters = model.parameters()
        else:
            dont_decay = dont_decay_weights if dont_decay_weights else ['layer_norm.weight', 'bias']
            params_w_wd = [p for n, p in model.named_parameters() if not any(nd in n for nd in dont_decay)]
            params_wo_wd = [p for n, p in model.named_parameters() if any(nd in n for nd in dont_decay)]
            parameters = [
                {'params': params_w_wd, 'weight_decay': weight_decay},
                {'params': params_wo_wd, 'weight_decay': 0.0},
            ]
        self.global_step = global_step
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.lr = lr
        self._decay = self._cosine_decay if decay_type == 'cosine' else self._linear_decay
        self.warmup_fract = warmup_fract
        self.plateau_fract = plateau_fract
        self.alpha = alpha_decay
        self.model = model
        self.loss_function = model.create_loss()
        self.device = 'cpu'
        self.num_train_workers = num_train_workers
        self.num_valid_workers = num_valid_workers
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
            self.loss_function = self.loss_function.to(self.device)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.collate_function = collate_function
        logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    def __str__(self):
        return '\n\t'.join(
            [self.__class__.__name__]
            + [f'{k}={v}' for k, v in self.__dict__.items() if v is not None and type(v) in [str, int, float]]
        )

    def compute_train_steps_per_epoch(self, dataset: Dataset) -> int:
        """Compute the number of steps in an epoch given the batch size and gradient accum

        :param dataset: A dataset consisting of unbatched data
        :return: The steps per epoch
        """
        steps_per_epoch = len(dataset) // self.batch_size
        return steps_per_epoch

    def train_steps(
            self,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            model_base: str,
            num_steps: int = 250_000,
            saves_per_cycle: int = 1,
            train_cycle_size: int = 10000,
            eval_cycle_size: int = 2500,
    ):
        """Train for a fixed number of steps using an infinite dataset

        We provide an iterable training set and evaluation set, and we pick a max number of steps.
        We determine a cycle length which is similar to an epoch in that we evaluate after each cycle.
        Saves are done per cycle now, similar to epochs.  This method is extremely flexible and can be used for
        very large datasets that might only be a single epoch per ~1M steps e.g.

        :param train_dataset: Infinite iterable dataset
        :param eval_dataset: Infinite iterable dataset
        :param model_base: Usually something like `/path/to/dir/ckpt` which will be suffixed when written
        :param num_steps: Total number of steps to train for
        :param saves_per_cycle: The number of times to save per train cycle
        :param train_cycle_size: Like a mini-epoch, we will train for a few steps, then run eval
        :param eval_cycle_size: Like a mini-epoch, we will eval for a few steps, then go back
        :return:
        """

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            collate_fn=self.collate_function,
            drop_last=True,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_valid_workers,
            collate_fn=self.collate_function,
            drop_last=True,
        )

        self.total_steps = num_steps
        save_iter = train_cycle_size // saves_per_cycle
        iters_left = (self.total_steps - self.global_step)
        current_cycle = 0 if self.global_step == 0 else (self.global_step // train_cycle_size)
        logging.info(
            'steps per cycle [%d], global step [%d], total train steps [%d] current cycle [%d], saves per cycle [%d]',
            train_cycle_size,
            self.global_step,
            self.total_steps,
            current_cycle,
            saves_per_cycle,
        )
        train_iter = iter(train_data_loader)
        eval_iter = iter(eval_data_loader)
        while self.global_step < self.total_steps:
            num_iters = min(iters_left, train_cycle_size)
            metrics = self._train_some(train_iter, num_iters, save_iter, model_base)
            # Log our metrics
            logging.info(metrics)

            metrics = self._eval_some(eval_iter, eval_cycle_size)
            logging.info(metrics)

    def show_lr_plan(self, total_steps: Optional[int] = None):
        """This function plots the learning regimen over a number of steps given

        If no the total_steps is not given, it uses what the trainer has set already

        It saves the existing steps, applies the total_steps as a field, and then
        runs

        :param total_steps:
        :return:
        """
        import matplotlib.pyplot as plt

        if total_steps is not None:
            save_total_steps = self.total_steps
            self.total_steps = total_steps
        steps = np.arange(0, self.total_steps)
        y = [self._lr_step(step) for step in steps]

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'Learning rate schedule for trainer with {total_steps} steps')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.plot(steps, y, label=f"steps={total_steps}")
        ax.plot(self.global_step, self._lr_step(self.global_step), 'go', label='last position')
        if total_steps is not None:
            self.total_steps = save_total_steps
        ax.legend()
        plt.show()

    def train_epochs(
            self,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            model_base: str,
            num_epochs: int = 1,
            saves_per_epoch: int = 1,
    ):
        """Epoch training interface for trainer for Map-style Datasets

        Under the hood, the trainer calculates the number of total steps you request, and sets up the learning
        schedule using that.  This epoch style approach makes sense when you are using basic data loaders that
        run over an epoch.

        :param train_dataset:  The dataset to train on
        :param eval_dataset: The dataset to evaluate on
        :param model_base: Usually something like `/path/to/dir/ckpt` which will be suffixed when written
        :param num_epochs: The number of times to run over the training set.  This gets converted to a total num steps
        :param saves_per_epoch: How many times should we save our dataset in the middle of an epoch, default=1
        :return:
        """
        # Total number of steps per epoch, taking into account grad accum
        steps_per_epoch = self.compute_train_steps_per_epoch(train_dataset)
        # The total number of steps for epoch training is always the number of epoch steps x the number of epochs
        self.total_steps = steps_per_epoch * num_epochs

        # If our model was already run, find the closest epoch and start from there, otherwise set to 0
        current_epoch = 0 if self.global_step == 0 else (self.global_step // steps_per_epoch) - 1

        logging.info(
            'steps per epoch [%d], global step [%d], total train steps [%d] current epoch [%d], saves per epoch [%d]',
            steps_per_epoch,
            self.global_step,
            self.total_steps,
            current_epoch,
            saves_per_epoch,
        )
        while current_epoch < num_epochs and self.global_step < self.total_steps:
            train_data_loader = DataLoader(
                train_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_train_workers,
                collate_fn=self.collate_function,
            )

            # For train some, we need figure out how many iters are needed
            total_iters_left = self.total_steps - self.global_step
            # If we restarted, we may have to run only a fraction of our epoch, so take the min between iters and whats
            # left to do
            num_iters = min(total_iters_left, len(train_data_loader))

            # We always save a fixed number of times per epoch.  If we restarted and there isnt much to do, we catch
            # that when we break out and save
            save_iter = len(train_data_loader) // saves_per_epoch
            # Train some steps using our iterator
            metrics = self._train_some(iter(train_data_loader), num_iters, save_iter, model_base)
            # Log our metrics
            logging.info(metrics)

            # For evaluation
            eval_data_loader = DataLoader(
                eval_dataset,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=self.num_valid_workers,
                collate_fn=self.collate_function,
            )
            num_iters = len(eval_data_loader)
            metrics = self._eval_some(iter(eval_data_loader), num_iters)
            logging.info(metrics)
            self._save_checkpoint(model_base)
            current_epoch += 1

    def _warmup(self, global_step):
        warmup_steps = self.warmup_fract * self.total_steps
        lr_factor = min(1.0, global_step / warmup_steps)
        return self.lr * lr_factor

    def _cosine_decay(self, global_step):
        global_step = min(global_step, self.total_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.total_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.lr * decayed

    def _linear_decay(self, global_step):
        global_step = min(global_step, self.total_steps)
        scaled_lr = self.lr * (1.0 - self.alpha) * (1.0 - global_step / self.total_steps) + (self.alpha * self.lr)
        return scaled_lr

    def _lr_step(self, global_step):
        total_steps_lr1 = self.total_steps * (self.warmup_fract + self.plateau_fract)
        if global_step < total_steps_lr1:
            return self._warmup(global_step)
        return self._decay(global_step - total_steps_lr1)

    def _train_some(self, data_iter, num_iters, save_iter, model_base):
        """Train for some number of steps with an iterator.  The iterator should have at least that many steps available

        This can be called externally as an epoch runner or just some number of fixed steps.
        We will define iteration here as when an iterator yields a batch of data.  We will distinguish this from a
        step, which could consist of multiple iterations if using gradient accumulation.

        :param data_iter: An iterator wrapping a DataLoader
        :param num_iters: The number of iterations to get from the data loader
        :param save_iter: How many iterations before we save a checkpoint
        :param model_base: The model base for writing checkpoints
        :return: The training metrics
        """
        avg_loss = Average('average_train_loss')
        metrics = {}
        self.optimizer.zero_grad()
        start = time.time()
        self.model.train()

        progress = tqdm(range(num_iters), total=num_iters)
        for iters in progress:
            (x, y) = next(data_iter)
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            logits = self.model(x)
            loss = self.loss_function(logits.reshape(-1, self.model.vocab_size), y.view(-1))
            loss.backward()
            avg_loss.update(loss.item())
            if (iters + 1) % save_iter == 0:
                self._save_checkpoint(model_base)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1
            if self.global_step == self.total_steps:
                progress.set_description(
                    f"global step {self.global_step}: loss {loss.item():.5f}. lr {self.current_lr:e}"
                )
                self._save_checkpoint(model_base)
                break
            self.optimizer.zero_grad()
            self.current_lr = self._lr_step(self.global_step)
            for p in self.optimizer.param_groups:
                p["lr"] = self.current_lr

            progress.set_description(
                f"global step {self.global_step}: loss {loss.item():.5f}. lr {self.current_lr:e}"
            )

        # How much time elapsed in minutes
        elapsed = (time.time() - start) / 60
        train_token_loss = avg_loss.avg
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        return metrics

    def _eval_some(self, data_iter, num_iters):
        """Evaluate some data

        :param data_iter: A data iterator
        :param num_iters: The number of iterations to train for
        :return: The validation metrics
        """
        self.model.eval()
        avg_loss = Average('average_valid_loss')
        start = time.time()
        metrics = {}
        progress = tqdm(range(num_iters), total=num_iters)
        with torch.no_grad():
            for iters in progress:
                (x, y) = next(data_iter)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                logits = self.model(x)
                loss = self.loss_function(logits.reshape(-1, self.model.vocab_size), y.view(-1))
                avg_loss.update(loss.item())
                progress.set_description(f"validation steps {iters}: loss {loss.item():.5f}. lr {self.current_lr:e}")
        valid_token_loss = avg_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start) / 60
        metrics['valid_elapsed_min'] = elapsed
        metrics['average_valid_loss'] = valid_token_loss
        metrics['average_valid_word_ppl'] = valid_token_ppl
        return metrics

    def _checkpoint_for(self, model_base):
        return f'{model_base}-step-{self.global_step}'

    def _save_checkpoint(self, model_base: str):
        checkpoint_name = self._checkpoint_for(model_base)
        logging.debug('Saving checkpoint [%s]', checkpoint_name)
        model_ = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_.state_dict(), checkpoint_name + '.pth')


def init_distributed(local_rank):
    if local_rank == -1:
        # https://github.com/kubeflow/pytorch-operator/issues/128
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        logger.info("Setting local rank to RANK env variable")
        local_rank = int(os.environ['RANK'])
    logger.warning("Local rank (%d)", local_rank)
    # In an env like k8s with kubeflow each worker will only see a single gpu
    # with an id of 0. If the gpu count is 1 then we are probably in an env like
    # that so we should just use the first (and only) gpu avaiable
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return device, local_rank


def get_num_gpus_multiworker() -> int:
    """Get the number of GPUs in multi-worker distributed training

    :return: A number of GPUs
    """
    return int(os.environ.get("WORLD_SIZE", 1))


class DistributedLMTrainer:
    """Trainer that works on multiple GPUs or machines with distributed data parallel

    This trainer assumes pure data parallelism -- each model is on a single gpu
    If we wanted to support model and data parallelism we would need to update
    the selection of gpus based on rank, it would need to select multiple ids
    based on rank, here we select only a single gpu and use it for input and
    output.

    """

    def __init__(
            self,
            model,
            lr: float = 1.0e-4,
            batch_size: int = 256,
            weight_decay: float = 1.0e-2,
            warmup_fract: int = 0.1,
            plateau_fract: int = 0.0,
            decay_type: str = 'cosine',
            total_steps: Optional[int] = None,
            global_step: int = 0,
            alpha_decay: float = 0.0,
            betas: Tuple[float] = (0.9, 0.98),
            eps=1e-08,
            grad_clip: float = 1.0,
            local_rank=-1,
            num_train_workers=4,
            num_valid_workers=1,
            dont_decay_weights: Optional[List[str]] = None,
            collate_function: Optional[Callable] = None,
            **kwargs,
    ):

        if weight_decay == 0.0:
            parameters = model.parameters()
        else:
            dont_decay = dont_decay_weights if dont_decay_weights else ['layer_norm.weight', 'bias']
            params_w_wd = [p for n, p in model.named_parameters() if not any(nd in n for nd in dont_decay)]
            params_wo_wd = [p for n, p in model.named_parameters() if any(nd in n for nd in dont_decay)]
            parameters = [
                {'params': params_w_wd, 'weight_decay': weight_decay},
                {'params': params_wo_wd, 'weight_decay': 0.0},
            ]
        self.global_step = global_step
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.lr = lr
        self._decay = self._cosine_decay if decay_type == 'cosine' else self._linear_decay
        self.warmup_fract = warmup_fract
        self.plateau_fract = plateau_fract
        self.alpha = alpha_decay
        self.model = model
        self.loss_function = model.create_loss()
        self.device = 'cpu'
        self.num_train_workers = num_train_workers
        self.num_valid_workers = num_valid_workers
        self.num_gpus = get_num_gpus_multiworker()
        self.device, self.local_rank = init_distributed(local_rank)
        logger.info(f"Using {self.num_gpus} GPUs in this job.")

        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)

        model = DistributedDataParallel(
            model, device_ids=[self.device], output_device=self.device, find_unused_parameters=True
        )
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.collate_function = collate_function
        logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    def __str__(self):
        return '\n\t'.join(
            [self.__class__.__name__]
            + [f'{k}={v}' for k, v in self.__dict__.items() if v is not None and type(v) in [str, int, float]]
        )

    def train_steps(
            self,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            model_base: str,
            num_steps: int = 250_000,
            saves_per_cycle: int = 1,
            train_cycle_size: int = 10000,
            eval_cycle_size: int = 2500,
            log_updates_per_train_cycle: Optional[int] = None,
    ):
        """Train for a fixed number of steps using Infinite datasets

        We provide an iterable training set and evaluation set, and we pick a max number of steps.
        We determine a cycle length which is similar to an epoch in that we evaluate after each cycle.
        Saves are done per cycle now, similar to epochs.  This method is extremely flexible and can be used for
        very large datasets that might only be a single epoch per ~1M steps e.g.

        :param train_dataset: Infinite iterable dataset
        :param eval_dataset: Infinite iterable dataset
        :param model_base: Usually something like `/path/to/dir/ckpt` which will be suffixed when written
        :param num_steps: Total number of steps to train for
        :param saves_per_cycle: The number of times to save per train cycle
        :param train_cycle_size: Like a mini-epoch, we will train for a few steps, then run eval
        :param eval_cycle_size: Like a mini-epoch, we will eval for a few steps, then go back
        :param log_updates_per_train_cycle: How many times to log our stats per cycle.  Defaults to 10x number of saves
        :return:
        """

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            collate_fn=self.collate_function,
            drop_last=True,
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_valid_workers,
            collate_fn=self.collate_function,
            drop_last=True,
        )

        self.total_steps = num_steps
        # On each machine, this still is right
        local_iters_per_cycle = train_cycle_size
        save_iter = local_iters_per_cycle // saves_per_cycle

        local_iters_left = (self.total_steps - self.global_step)
        current_cycle = 0 if self.global_step == 0 else (self.global_step // train_cycle_size)
        logging.info(
            'steps per cycle [%d], global step [%d], total train steps [%d] current cycle [%d], saves per cycle [%d]',
            train_cycle_size,
            self.global_step,
            self.total_steps,
            current_cycle,
            saves_per_cycle,
        )
        train_iter = iter(train_data_loader)
        eval_iter = iter(eval_data_loader)
        while self.global_step < self.total_steps:
            num_iters = min(local_iters_left, local_iters_per_cycle)
            metrics = self._train_some(train_iter, num_iters, save_iter, model_base, log_updates_per_train_cycle)
            # Log our metrics
            logging.info(metrics)
            if self.local_rank < 1:
                metrics = self._eval_some(eval_iter, eval_cycle_size)
            logging.info(metrics)

    def show_lr_plan(self, total_steps: Optional[int] = None):
        """This function plots the learning regimen over a number of steps given

        If no the total_steps is not given, it uses what the trainer has set already

        It saves the existing steps, applies the total_steps as a field, and then
        runs

        :param total_steps:
        :return:
        """
        import matplotlib.pyplot as plt

        if total_steps is not None:
            save_total_steps = self.total_steps
            self.total_steps = total_steps
        steps = np.arange(0, self.total_steps)
        y = [self._lr_step(step) for step in steps]

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'Learning rate schedule for trainer with {total_steps} steps')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.plot(steps, y, label=f"steps={total_steps}")
        ax.plot(self.global_step, self._lr_step(self.global_step), 'go', label='last position')
        if total_steps is not None:
            self.total_steps = save_total_steps
        ax.legend()
        plt.show()

    def _warmup(self, global_step):
        warmup_steps = self.warmup_fract * self.total_steps
        lr_factor = min(1.0, global_step / warmup_steps)
        return self.lr * lr_factor

    def _cosine_decay(self, global_step):
        global_step = min(global_step, self.total_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.total_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.lr * decayed

    def _linear_decay(self, global_step):
        global_step = min(global_step, self.total_steps)
        scaled_lr = self.lr * (1.0 - self.alpha) * (1.0 - global_step / self.total_steps) + (self.alpha * self.lr)
        return scaled_lr

    def _lr_step(self, global_step):
        total_steps_lr1 = self.total_steps * (self.warmup_fract + self.plateau_fract)
        if global_step < total_steps_lr1:
            return self._warmup(global_step)
        return self._decay(global_step - total_steps_lr1)

    def _train_some(self, data_iter, num_iters, save_iter, model_base, update_iter=None):
        """Train for some number of steps with an iterator.  The iterator should have at least that many steps available

        This can be called externally as an epoch runner or just some number of fixed steps.
        We will define iteration here as when an iterator yields a batch of data.  We will distinguish this from a
        step, which could consist of multiple iterations if using gradient accumulation.

        :param data_iter: An iterator wrapping a DataLoader
        :param num_iters: The number of iterations to get from the data loader
        :param save_iter: How many iterations before we save a checkpoint
        :param model_base: The model base for writing checkpoints
        :param update_iter: If None, this will default to 10 updates per checkpoint
        :return: The training metrics
        """
        avg_loss = Average('average_train_loss')
        metrics = {}
        if update_iter is None:
            update_iter = save_iter // 10
        # Note that the gradients are zero'd here.  This is why we want our cycle_steps to be a multiple of grad_accum

        start = time.time()
        self.model.train()
        for iters in range(num_iters):

            self.optimizer.zero_grad()
            (x, y) = next(data_iter)
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            logits = self.model(x)
            loss = self.loss_function(logits.reshape(-1, self.model.vocab_size), y.view(-1))
            loss.backward()
            avg_loss.update(loss.item())

            if (iters + 1) % update_iter == 0:
                logger.info(
                    f"global step {self.global_step}: loss {avg_loss}. lr {self.current_lr:e}"
                )

            # Only save on master
            if self.local_rank < 1 and (iters + 1) % save_iter == 0:
                self._save_checkpoint(model_base)

            self.optimizer.step()
            self.global_step += 1
            if self.global_step == self.total_steps:
                logger.info(
                    f"global step {self.global_step}: loss {loss.item():.5f}. lr {self.current_lr:e}"
                )
                if self.local_rank < 1:
                    self._save_checkpoint(model_base)
                break

            self.current_lr = self._lr_step(self.global_step)
            for p in self.optimizer.param_groups:
                p["lr"] = self.current_lr

        # How much time elapsed in minutes
        elapsed = (time.time() - start) / 60
        train_token_loss = avg_loss.avg
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        return metrics

    def _eval_some(self, data_iter, num_iters):
        """Evaluate some data

        :param data_iter: A data iterator
        :param num_iters: The number of iterations to train for
        :return: The validation metrics
        """
        self.model.eval()
        avg_loss = Average('average_valid_loss')
        start = time.time()
        metrics = {}
        progress = tqdm(range(num_iters), total=num_iters)
        with torch.no_grad():
            for iters in progress:
                (x, y) = next(data_iter)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                logits = self.model(x)
                loss = self.loss_function(logits.reshape(-1, self.model.vocab_size), y.view(-1))
                avg_loss.update(loss.item())
                progress.set_description(f"validation steps {iters}: loss {loss.item():.5f}. lr {self.current_lr:e}")
        valid_token_loss = avg_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start) / 60
        metrics['valid_elapsed_min'] = elapsed
        metrics['average_valid_loss'] = valid_token_loss
        metrics['average_valid_word_ppl'] = valid_token_ppl
        return metrics

    def _checkpoint_for(self, model_base):
        return f'{model_base}-step-{self.global_step}'

    def _save_checkpoint(self, model_base: str):
        checkpoint_name = self._checkpoint_for(model_base)
        logging.debug('Saving checkpoint [%s]', checkpoint_name)
        model_ = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_.state_dict(), checkpoint_name + '.pth')


if __name__ == '__main__':
    pass
