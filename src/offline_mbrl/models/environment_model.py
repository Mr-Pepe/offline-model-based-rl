#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains an environment model that can be used to generate data for RL agents."""


import os
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import torch
from ray import tune
from torch import nn
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter
from torch.optim.adamw import AdamW

from offline_mbrl.models.multi_head_mlp import MultiHeadMlp
from offline_mbrl.utils.modes import (
    ALEATORIC_PARTITIONING,
    ALEATORIC_PENALTY,
    ALL_MODES,
    EPISTEMIC_PARTITIONING,
    EPISTEMIC_PENALTY,
    PARTITIONING_MODES,
    PENALTY_MODES,
)
from offline_mbrl.utils.replay_buffer import ReplayBuffer


class EnvironmentModel(nn.Module):
    """An environment model that can be used to generate training data for RL agents.

    The model uses an ensemble of neural networks for prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layer_sizes: tuple[int, ...] = (128, 128),
        type: Literal[  # pylint: disable=redefined-builtin
            "deterministic", "probabilistic"
        ] = "deterministic",
        n_networks: int = 1,
        device: str = "cpu",
        preprocessing_function: Optional[Callable] = None,
        termination_function: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        obs_bounds_trainable: bool = True,
        reward_bounds_trainable: bool = True,
        **unused_kwargs: dict,
    ):
        """Initializes the environment model.

        Assumes that observations are flattened to one dimension.

        Args:
            obs_dim (int): Observation size.
            act_dim (int): Action size.
            hidden_layer_sizes (tuple[int, ...], optional): The number of neurons in
                each hidden layer of each neural network in the environment model.
                Defaults to (128, 128).
            type (Literal["deterministic", "probabilistic"], optional): The model type.
                A probabilistic model will predict a mean and variance for each output
                value and create a sample, assuming a normal distribution.
                Defaults to "deterministic".
            n_networks (str, optional): The number of networks to use in the ensemble.
                Defaults to 1.
            device (str, optional): The device to push the model to. Defaults to "cpu".
            preprocessing_function (Optional[Callable], optional): A function that
                transforms the model input while preserving the shape. Defaults to None.
            termination_function (Optional[Callable], optional): A function that
                determines whether states predicted by the model are terminal states or
                not. The predicted terminal signal will always be False if no
                termination function is provided. Defaults to None.
            reward_function (Optional[Callable], optional): A custom reward function
                that will be used instead of the predicted reward. Defaults to None.
            obs_bounds_trainable (bool, optional): I forgot what this thing does. Tell
                me if you find out. Defaults to True.
            reward_bounds_trainable (bool, optional): See :code:`obs_bounds_trainable`.
                Defaults to True.

        Raises:
            ValueError: If the provided type is not valid.
        """
        super().__init__()

        self.model_type = type

        if type not in ("deterministic", "probabilistic"):
            raise ValueError(f"Unknown model type '{type}'")

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Append reward signal
        self.out_dim = obs_dim + 1

        self.n_networks = n_networks
        self.preprocessing_function = preprocessing_function
        self.termination_function = termination_function
        self.reward_function = reward_function

        self.layers = MultiHeadMlp(obs_dim, act_dim, hidden_layer_sizes, n_networks)

        # Taken from https://github.com/kchua/handful-of-trials/blob/master/
        # dmbrl/modeling/models/BNN.py
        self.obs_max_logvar = Parameter(
            torch.ones(n_networks, (self.obs_dim)) * 0.5,
            requires_grad=obs_bounds_trainable,
        )

        self.rew_max_logvar = Parameter(
            torch.ones(n_networks, (1)) * 1, requires_grad=reward_bounds_trainable
        )

        self.min_logvar = Parameter(
            torch.ones(n_networks, (self.out_dim)) * -10,
            requires_grad=obs_bounds_trainable,
        )

        self.to(device)

        self.max_logvar = torch.cat((self.obs_max_logvar, self.rew_max_logvar), dim=1)

        self.optim: Optional[AdamW] = None

        self.has_been_trained_at_least_once = False

        self.max_obs_act = None
        self.min_obs_act = None
        self.max_reward: Optional[torch.Tensor] = None

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Env model parameters: {n_params}")

    def forward(
        self, raw_obs_act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts the next observation, reward and terminal signal.

        This method is used during model training.
        Use the :py:meth:`.get_prediction` method to retrieve predictions to generate
        data for agent training.

        Args:
            raw_obs_act (torch.Tensor): A batch of concatenated observations and
                actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The predicted next observation, reward and terminal signal as a tuple of
                the actual predicted values, the predicted mean values, the predicted
                log variances, maximum log variances, and the minimum log variances.
        """
        device = next(self.layers.parameters()).device
        raw_obs_act = self._adjust_device_and_shape_if_necessary(raw_obs_act, device)

        if self.preprocessing_function:
            obs_act = self.preprocessing_function(raw_obs_act)
        else:
            obs_act = raw_obs_act.detach().clone()

        predicted_obs_deltas, predicted_rewards = self.layers(obs_act)

        predicted_next_obs = (
            predicted_obs_deltas[:, :, : self.obs_dim] + raw_obs_act[:, : self.obs_dim]
        )

        means = torch.cat(
            (
                predicted_next_obs,
                predicted_rewards[:, :, 0].view((self.n_networks, -1, 1)),
            ),
            dim=2,
        )

        if self.model_type == "deterministic":

            predictions = means
            logvars = torch.zeros(
                (self.n_networks, obs_act.shape[0], obs_act.shape[1] + 1)
            )

        else:
            logvars = torch.cat(
                (
                    predicted_obs_deltas[:, :, self.obs_dim :],
                    predicted_rewards[:, :, 1].view((self.n_networks, -1, 1)),
                ),
                dim=2,
            )

            max_logvar = self.max_logvar.unsqueeze(1)
            min_logvar = self.min_logvar.unsqueeze(1)
            logvars = max_logvar - softplus(max_logvar - logvars)
            logvars = min_logvar + softplus(logvars - min_logvar)

            std = torch.exp(0.5 * logvars)

            predictions = torch.normal(means, std)

        return (
            predictions,
            means,
            logvars,
            self.max_logvar,
            self.min_logvar,
        )

    def get_prediction(
        self,
        raw_obs_act: torch.Tensor,
        i_network: int = -1,
        pessimism: float = 0,
        mode: Optional[str] = None,
        ood_threshold: float = 10000,
        debug: bool = False,
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Predicts the next observation, reward, and terminal signal.

        Args:
            raw_obs_act (torch.Tensor): A batch of concatenated observations and
                actions.
            i_network (int, optional): The network to use for prediction. Set to -1 to
                get a prediction from a random network. Defaults to -1.
            pessimism (float, optional): The amount of pessimism to use for predicting
                the reward and terminal signal. Defaults to 0.
            mode (Optional[str], optional): See :py:mod:`.modes`. Defaults to None.
            ood_threshold (float, optional): The out-of-distribution to use for
                partitioning modes. Defaults to 10000.
            debug (bool, optional): Detailed information is provided alongside the
                prediction if this flag is set. Defaults to False.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor]]: The prediction + optional debug
                information if the debug argument is set to True.
        """
        device = next(self.layers.parameters()).device

        if i_network == -1:
            i_network = int(torch.randint(self.n_networks, (1,)).item())

        self.eval()

        with torch.no_grad():
            predictions, means, logvars, _, _ = self.forward(raw_obs_act)

        predicted_next_obs = predictions[:, :, :-1]

        if self.termination_function:
            dones = self.termination_function(predicted_next_obs).to(device)
        else:
            dones = torch.zeros(
                (self.n_networks, raw_obs_act.shape[0], 1), device=device
            )

        if self.reward_function:
            predicted_rewards = self.reward_function(
                obs=raw_obs_act[:, : self.obs_dim],
                act=raw_obs_act[:, self.obs_dim :],
                next_obs=predicted_next_obs,
            )
        else:
            predicted_rewards = predictions[:, :, -1].unsqueeze(-1)

        predictions = torch.cat((predicted_next_obs, predicted_rewards, dones), dim=2)
        prediction = predictions[i_network]

        if self.preprocessing_function is not None:
            preprocessed_means = self.preprocessing_function(
                means[:, :, :-1], detach=False
            )
        else:
            preprocessed_means = means[:, :, :-1]

        epistemic_uncertainty = (
            torch.cdist(
                torch.transpose(preprocessed_means, 0, 1),
                torch.transpose(preprocessed_means, 0, 1),
            )
            .max(-1)
            .values.max(-1)
            .values
        )

        aleatoric_uncertainty = (
            torch.exp(logvars[:, :, :-1]).max(dim=0).values.max(dim=1).values.to(device)
        )

        if mode is not None:
            self._check_mode_is_valid(mode)

            if mode in PENALTY_MODES:
                if mode == ALEATORIC_PENALTY:
                    uncertainty = aleatoric_uncertainty
                elif mode == EPISTEMIC_PENALTY:
                    uncertainty = epistemic_uncertainty

                # Penalize the predicted reward
                prediction[:, -2] = (
                    means[:, :, -1].mean(dim=0) - pessimism * uncertainty
                )

            elif mode in PARTITIONING_MODES:
                if mode == ALEATORIC_PARTITIONING:
                    ood_idx = aleatoric_uncertainty > ood_threshold
                elif mode == EPISTEMIC_PARTITIONING:
                    ood_idx = epistemic_uncertainty > ood_threshold

                assert self.max_reward is not None
                # Set reward to negative maximum reward for out-of-distribution samples
                # pylint: disable=invalid-unary-operand-type
                prediction[ood_idx, -2] = -self.max_reward
                # Set done signal
                prediction[ood_idx, -1] = 1

        if debug:
            return (
                prediction,
                means,
                logvars,
                epistemic_uncertainty,
                aleatoric_uncertainty,
            )

        return prediction

    def _adjust_device_and_shape_if_necessary(
        self, x: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x

    def train_to_convergence(
        self,
        replay_buffer: ReplayBuffer,
        lr: float = 1e-3,
        batch_size: int = 1024,
        val_split: float = 0.2,
        patience: int = 20,
        debug: bool = False,
        max_n_train_batches: int = -1,
        max_n_train_epochs: int = -1,
        checkpoint_dir: Optional[Path] = None,
        tuning: bool = False,
        **unused_kwargs: dict,
    ) -> tuple[torch.Tensor, int]:
        """Trains the environment model to convergence.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample from.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            batch_size (int, optional): The batch size. Defaults to 1024.
            val_split (float, optional): The proportion of samples from the replay
                buffer to use as validation set. Defaults to 0.2.
            patience (int, optional): The training is considered converged when the
                validation loss has not decreased for this many epochs. Defaults to 20.
            debug (bool, optional): Whether to print debug information during training.
                Defaults to False.
            max_n_train_batches (int, optional): Training stops after training on this
                many batches. Defaults to -1.
            max_n_train_epochs (int, optional): Training stops after training for this
                many epochs. Defaults to -1.
            checkpoint_dir (Optional[Path], optional): A path to save checkpoints of the
                model to during a Ray Tune run. Defaults to None.
            tuning (bool, optional): Whether the model is trained as part of a Ray Tune
                run. Defaults to False.

        Raises:
            ValueError: If the replay buffer is not big enough to provide a
                training/validation split with the provided batch size.

        Returns:
            tuple[torch.Tensor, int]: The final validation loss for each network in the
                ensemble and the number of batches that the model for trained on.
        """
        n_train_batches = int((replay_buffer.size * (1 - val_split)) // batch_size)
        n_val_batches = int((replay_buffer.size * val_split) // batch_size)

        if n_train_batches == 0 or n_val_batches == 0:
            raise ValueError(
                f"Dataset of size {replay_buffer.size} not big enough to "
                f"generate a {val_split*100} % "
                f"validation split with batch size {batch_size}."
            )

        print("")
        print("Training environment model to convergence")
        print(
            f"Buffer size: {replay_buffer.size} "
            f"Train batches per epoch: {n_train_batches} "
            f"Stopping after {max_n_train_batches} batches"
        )

        device = next(self.parameters()).device
        use_amp = "cuda" in device.type
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp, growth_factor=1.5, backoff_factor=0.7
        )

        if self.optim is None:
            self.optim = AdamW(self.parameters(), lr=lr)

        assert self.optim is not None

        if checkpoint_dir:
            print(f"Loading environment model from checkpoint '{checkpoint_dir}'.")
            checkpoint = torch.load(checkpoint_dir / "checkpoint")
            self.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["step"]
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

        self.train()

        min_val_loss = 1e10
        epochs_since_last_performance_improvement = 0
        batches_trained = 0

        stop_training = False

        val_loss_per_network = torch.zeros((self.n_networks))

        epoch = 0

        while (
            epochs_since_last_performance_improvement < patience
            and (max_n_train_epochs == -1 or epoch < max_n_train_epochs)
            and not stop_training
        ):

            total_train_loss = 0.0

            for _ in range(n_train_batches):
                train_loss = self.train_one_batch(
                    replay_buffer, val_split, batch_size, scaler, use_amp, debug
                )

                total_train_loss += train_loss

                if max_n_train_batches != -1 and max_n_train_batches <= batches_trained:
                    stop_training = True
                    break

                batches_trained += 1

            if debug:
                print("")
                print(f"Average training loss: {total_train_loss / n_train_batches}")

            avg_val_loss, val_loss_per_network = self.compute_validation_losses(
                replay_buffer, val_split, n_val_batches, batch_size
            )

            if avg_val_loss < min_val_loss:
                epochs_since_last_performance_improvement = 0
                min_val_loss = avg_val_loss
            else:
                epochs_since_last_performance_improvement += 1

            if tuning:
                self._save_model_checkpoint(epoch, avg_val_loss)

            epoch += 1

            print(
                f"Train batches: {batches_trained} "
                f"Patience: {epochs_since_last_performance_improvement}/{patience} "
                f"Val losses: {val_loss_per_network.tolist()}",
                end="\r",
            )

            if debug:
                print("")

        self.has_been_trained_at_least_once = True
        return val_loss_per_network, batches_trained

    def train_one_batch(
        self,
        replay_buffer: ReplayBuffer,
        val_split: float,
        batch_size: int,
        scaler: torch.cuda.amp.GradScaler,
        use_amp: bool,
        debug: bool,
    ) -> float:
        """Trains the model on a single batch sampled from a replay buffer.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample from.
            val_split (float): The proportion of samples from the replay buffer to use
                as validation set.
            batch_size (int): The batch size.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler to use.
            use_amp (bool): Whether or not to use mixed precision.
            debug (bool): Whether or not to print additional debug information.

        Returns:
            float: The training loss on the single batch.
        """
        device = next(self.parameters()).device

        model_input, ground_truth = get_model_input_and_ground_truth_from_batch(
            replay_buffer.sample_train_batch(batch_size, val_split), device
        )

        if self.max_reward is None:
            self.max_reward = ground_truth[:, -1].max()
        else:
            self.max_reward = torch.max(self.max_reward, ground_truth[:, -1].max())

        assert self.optim is not None
        self.optim.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            if self.model_type == "deterministic":
                loss = self.deterministic_loss(model_input, ground_truth)
            else:
                loss = self.probabilistic_loss(
                    model_input,
                    ground_truth,
                    preprocessing_function=self.preprocessing_function,
                    debug=debug,
                )

                if self.max_obs_act is None:
                    self.max_obs_act = model_input.max(dim=0).values
                else:
                    self.max_obs_act += (
                        model_input.max(dim=0).values - self.max_obs_act
                    ) * 0.001

                if self.min_obs_act is None:
                    self.min_obs_act = model_input.min(dim=0).values
                else:
                    self.min_obs_act += (
                        model_input.min(dim=0).values - self.min_obs_act
                    ) * 0.001

        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(self.optim)
        scaler.update()

        return loss.item()

    def compute_validation_losses(
        self,
        replay_buffer: ReplayBuffer,
        val_split: float,
        n_val_batches: int,
        batch_size: int,
    ) -> tuple[float, torch.Tensor]:
        """Computes the validation losses for all networks in the ensemble.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample from.
            val_split (float): The proportion of samples from the replay buffer to use
                as validation set.
            n_val_batches (int): The number of validation batches to sample.
            batch_size (int): The batch size.

        Returns:
            tuple[float, torch.Tensor]: The average validation loss across all networks
                and the validation loss per network.
        """
        device = next(self.parameters()).device
        validation_losses_per_network = torch.zeros((self.n_networks))

        for i_network in range(self.n_networks):
            for _ in range(n_val_batches):
                (
                    model_input,
                    ground_truth,
                ) = get_model_input_and_ground_truth_from_batch(
                    replay_buffer.sample_val_batch(batch_size, val_split), device
                )

                if self.model_type == "deterministic":
                    validation_losses_per_network[i_network] += self.deterministic_loss(
                        model_input, ground_truth, i_network
                    ).item()
                else:
                    validation_losses_per_network[i_network] += self.probabilistic_loss(
                        model_input,
                        ground_truth,
                        i_network,
                        only_mse=True,
                        preprocessing_function=self.preprocessing_function,
                    ).item()

            validation_losses_per_network[i_network] /= n_val_batches

        average_validation_loss = validation_losses_per_network.mean()

        return average_validation_loss.item(), validation_losses_per_network

    def _save_model_checkpoint(self, epoch: int, avg_val_loss: float) -> None:
        tune.report(val_loss=avg_val_loss)

        with tune.checkpoint_dir(step=epoch) as tune_checkpoint_dir:
            path = os.path.join(tune_checkpoint_dir, "checkpoint")

            assert self.optim is not None

            torch.save(
                {
                    "step": epoch,
                    "model_state_dict": self.state_dict(),
                    "optim_state_dict": self.optim.state_dict(),
                    "val_loss": avg_val_loss,
                },
                path,
            )

    def _check_mode_is_valid(self, mode: str) -> None:
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown mode: {mode}")

        if (
            mode in (ALEATORIC_PENALTY, ALEATORIC_PARTITIONING)
            and self.model_type == "deterministic"
        ):
            raise ValueError(
                "Can not use aleatoric methods with deterministic ensemble."
            )

    def deterministic_loss(
        self,
        model_input: torch.Tensor,
        ground_truth_output: torch.Tensor,
        i_network: int = -1,
    ) -> torch.Tensor:
        """Computes the deterministic mean squared error.

        Args:
            model_input (torch.Tensor): The model input.
            ground_truth_output (torch.Tensor): The ground truth output to compute the
                loss against.
            i_network (int, optional): The network to compute the loss for. Pass -1 to
                compute the mean loss across all networks of the model. Defaults to -1.

        Returns:
            torch.Tensor: The deterministic loss.
        """
        predicted_output, _, _, _, _ = self(model_input)

        if i_network == -1:
            return torch.square(
                predicted_output - ground_truth_output.unsqueeze(0)
            ).mean()

        return torch.square(
            predicted_output[i_network] - ground_truth_output.unsqueeze(0)
        ).mean()

    def probabilistic_loss(
        self,
        model_input: torch.Tensor,
        ground_truth_output: torch.Tensor,
        i_network: int = -1,
        only_mse: bool = False,
        preprocessing_function: Callable = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """Computes the probabilistic loss.

        Args:
            model_input (torch.Tensor): The model input.
            ground_truth_output (torch.Tensor): The ground truth output to compute the
                loss against.
            i_network (int, optional): The network to compute the loss for. Pass -1 to
                compute the mean loss across all networks of the model. Defaults to -1.
            only_mse (bool, optional): Returns only the mean squared error of the means.
                Defaults to False.
            preprocessing_function (Callable, optional): The prediction and ground truth
                are preprocessed before computing the loss if a preprocessing function
                is provided. Defaults to None.
            debug (bool, optional): Whether or not to print additional debug
                information. Defaults to False.

        Returns:
            torch.Tensor: The probabilistic loss.
        """
        _, mean, logvar, max_logvar, min_logvar = self(model_input)

        if preprocessing_function is not None:
            mean[:, :, :-1] = preprocessing_function(mean[:, :, :-1], detach=False)
            ground_truth_output[:, :-1] = preprocessing_function(
                ground_truth_output[:, :-1]
            )

        if i_network > -1:
            mean = mean[i_network]
            logvar = logvar[i_network]
            max_logvar = max_logvar[i_network]
            min_logvar = min_logvar[i_network]

        inv_var = torch.exp(-logvar)

        if only_mse:
            return torch.square(mean - ground_truth_output).mean()

        mse_loss = torch.square(mean - ground_truth_output)
        mse_inv_var_loss = (mse_loss * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.mean() - 0.01 * min_logvar.mean()

        if debug:
            assert self.optim is not None
            print(
                f"LR: {self.optim.param_groups[0]['lr']:.5f}, "
                f"State MSE: {mse_loss[:, :, :-1].mean().item():.5f}, "
                f"Rew MSE: {mse_loss[:, :, -1].mean().item():.5f}, "
                f"MSE + INV VAR: {mse_inv_var_loss.item():.5f} "
                f"VAR: {var_loss.item():.5f}, "
                f"BOUNDS: {var_bound_loss.item():.5f}, "
                f"MAX LOGVAR: {max_logvar.mean().item():.5f}, "
                f"MIN LOGVAR: {min_logvar.mean().item():.5f}, ",
                end="\r",
            )

        return mse_inv_var_loss + var_loss + var_bound_loss


def get_model_input_and_ground_truth_from_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the model inputs and ground truth outputs from a batch.

    Args:
        batch (dict[str, torch.Tensor]): A batch sampled from a
            :py:class:`.ReplayBuffer`.
        device (torch.device): The device to push the tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The model input and ground truth output from
            a batch.
    """
    model_input = torch.cat((batch["obs"], batch["act"]), dim=1)
    ground_truth = torch.cat((batch["next_obs"], batch["rew"].unsqueeze(1)), dim=1)

    model_input = model_input.to(device)
    ground_truth = ground_truth.to(device)

    return model_input, ground_truth
