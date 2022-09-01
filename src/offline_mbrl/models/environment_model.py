import os
from typing import Callable, Optional

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
    """Takes in a state and action and predicts next state and reward."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(128, 128),
        type="deterministic",  # pylint: disable=redefined-builtin
        n_networks=1,
        device="cpu",
        pre_fn=None,
        post_fn=None,
        rew_fn=None,
        use_batch_norm=False,
        obs_bounds_trainable=True,
        r_bounds_trainable=True,
        **_,
    ):
        """
        type (string): deterministic or probabilistic

        n_networks (int): number of networks in the ensemble
        """
        super().__init__()

        self.type = type
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Append reward signal
        self.out_dim = obs_dim + 1

        self.n_networks = n_networks
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.rew_fn = rew_fn

        if type not in ("deterministic", "probabilistic"):
            raise ValueError(f"Unknown type {type}")

        self.layers = MultiHeadMlp(
            obs_dim, act_dim, hidden, n_networks, use_batch_norm=use_batch_norm
        )

        # Taken from https://github.com/kchua/handful-of-trials/blob/master/
        # dmbrl/modeling/models/BNN.py
        self.obs_max_logvar = Parameter(
            torch.ones(n_networks, (self.obs_dim)) * 0.5,
            requires_grad=obs_bounds_trainable,
        )

        self.rew_max_logvar = Parameter(
            torch.ones(n_networks, (1)) * 1, requires_grad=r_bounds_trainable
        )

        self.min_logvar = Parameter(
            torch.ones(n_networks, (self.out_dim)) * -10,
            requires_grad=obs_bounds_trainable,
        )

        self.to(device)

        self.max_logvar = torch.cat((self.obs_max_logvar, self.rew_max_logvar), dim=1)

        self.optim = None

        self.has_been_trained_at_least_once = False

        self.max_obs_act = None
        self.min_obs_act = None
        self.max_reward = None

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Env model parameters: {n_params}")

    def forward(self, raw_obs_act):

        device = next(self.layers.parameters()).device
        raw_obs_act = self.check_device_and_shape(raw_obs_act, device)

        if self.pre_fn:
            obs_act = self.pre_fn(raw_obs_act)
        else:
            obs_act = raw_obs_act.detach().clone()

        pred_obs_deltas, pred_rewards, uncertainty = self.layers(obs_act)

        pred_next_obs = (
            pred_obs_deltas[:, :, : self.obs_dim] + raw_obs_act[:, : self.obs_dim]
        )

        means = torch.cat(
            (pred_next_obs, pred_rewards[:, :, 0].view((self.n_networks, -1, 1))), dim=2
        )

        if self.type == "deterministic":

            predictions = means
            logvars = torch.zeros(
                (self.n_networks, obs_act.shape[0], obs_act.shape[1] + 1)
            )
            uncertainty = torch.zeros_like(uncertainty)

        else:
            next_obs_logvars = pred_obs_deltas[:, :, self.obs_dim :]
            reward_logvars = pred_rewards[:, :, 1].view((self.n_networks, -1, 1))
            logvars = torch.cat((next_obs_logvars, reward_logvars), dim=2)

            self.max_logvar = torch.cat(
                (self.obs_max_logvar, self.rew_max_logvar), dim=1
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
            uncertainty,
        )

    def get_prediction(
        self,
        raw_obs_act,
        i_network=-1,
        pessimism=0,
        mode: Optional[str] = None,
        ood_threshold=10000,
        with_uncertainty=False,
        debug=False,
    ):

        device = next(self.layers.parameters()).device

        if i_network == -1:
            i_network = torch.randint(self.n_networks, (1,)).item()

        self.eval()

        with torch.no_grad():
            predictions, means, logvars, _, _, explicit_uncertainties = self.forward(
                raw_obs_act
            )

        pred_next_obs = predictions[:, :, :-1]
        pred_rewards = predictions[:, :, -1].unsqueeze(-1)

        dones = None

        if self.post_fn:
            post = self.post_fn(next_obs=pred_next_obs, means=means, logvars=logvars)

            if "dones" in post:
                dones = post["dones"].to(device)
            if "means" in post:
                means = post["means"].to(device)
            if "logvars" in post:
                logvars = post["logvars"].to(device)

        if self.rew_fn:
            pred_rewards = self.rew_fn(
                obs=raw_obs_act[:, : self.obs_dim],
                act=raw_obs_act[:, self.obs_dim :],
                next_obs=pred_next_obs,
            )

        if dones is None:
            dones = torch.zeros(
                (self.n_networks, raw_obs_act.shape[0], 1), device=device
            )

        predictions = torch.cat((pred_next_obs, pred_rewards, dones), dim=2)
        prediction = predictions[i_network]

        if self.pre_fn is not None:
            norm_means = self.pre_fn(means[:, :, :-1], detach=False)
        else:
            norm_means = means[:, :, :-1]

        epistemic_uncertainty = (
            torch.cdist(
                torch.transpose(norm_means, 0, 1), torch.transpose(norm_means, 0, 1)
            )
            .max(-1)
            .values.max(-1)
            .values
        )

        aleatoric_uncertainty = (
            torch.exp(logvars[:, :, :-1]).max(dim=0).values.max(dim=1).values.to(device)
        )

        explicit_uncertainty = explicit_uncertainties[:, :, -1].mean(dim=0)

        underestimated_reward = pred_rewards.min(dim=0).values.view(-1)

        if self.max_reward is not None:
            underestimated_reward = torch.clamp_min(
                underestimated_reward,
                -self.max_reward  # pylint: disable=invalid-unary-operand-type
                * 1.00001,
            )

        if mode is not None:

            self.check_prediction_arguments(mode, pessimism)

            if mode in PENALTY_MODES:
                if mode == ALEATORIC_PENALTY:
                    uncertainty = aleatoric_uncertainty
                elif mode == EPISTEMIC_PENALTY:
                    uncertainty = epistemic_uncertainty

                prediction[:, -2] = (
                    means[:, :, -1].mean(dim=0) - pessimism * uncertainty
                )

            elif mode in PARTITIONING_MODES:
                if mode == ALEATORIC_PARTITIONING:
                    ood_idx = aleatoric_uncertainty > ood_threshold
                elif mode == EPISTEMIC_PARTITIONING:
                    ood_idx = epistemic_uncertainty > ood_threshold

                prediction[
                    ood_idx, -2
                ] = -self.max_reward  # pylint: disable=invalid-unary-operand-type
                prediction[ood_idx, -1] = 1

        if debug:
            return (
                prediction,
                means,
                logvars,
                explicit_uncertainties,
                epistemic_uncertainty,
                aleatoric_uncertainty,
                underestimated_reward,
            )
        if with_uncertainty:
            return prediction, explicit_uncertainty

        return prediction

    def check_device_and_shape(self, x, device):
        x = x.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return x

    def train_to_convergence(
        self,
        data: ReplayBuffer,
        lr=1e-3,
        batch_size=1024,
        val_split=0.2,
        patience: int = 20,
        debug=False,
        max_n_train_batches=-1,
        no_reward=False,
        max_n_train_epochs=-1,
        checkpoint_dir=None,
        tuning=False,
        **_,
    ):

        n_train_batches = int((data.size * (1 - val_split)) // batch_size)
        n_val_batches = int((data.size * val_split) // batch_size)

        if n_train_batches == 0 or n_val_batches == 0:
            raise ValueError(
                f"Dataset of size {data.size} not big enough to "
                f"generate a {val_split*100} % "
                f"validation split with batch size {batch_size}."
            )

        print("")
        print(
            f"Buffer size: {data.size} "
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

        if checkpoint_dir:
            print("Loading from checkpoint.")
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["step"]
            self.optim.load_state_dict(checkpoint["optim_state_dict"])

        self.train()

        min_val_loss = 1e10
        n_bad_val_losses = 0
        avg_val_loss = 0
        avg_train_loss = 0

        batches_trained = 0

        stop_training = False

        avg_val_losses = torch.zeros((self.n_networks))

        epoch = 0

        while n_bad_val_losses < patience and (
            max_n_train_epochs == -1 or epoch < max_n_train_epochs
        ):

            avg_train_loss = 0

            for _ in range(n_train_batches):
                model_input, ground_truth = get_model_input_and_ground_truth_from_batch(
                    data.sample_train_batch(batch_size, val_split), device
                )

                if self.max_reward is None:
                    self.max_reward = ground_truth[:, -1].max()
                else:
                    self.max_reward = torch.max(
                        self.max_reward, ground_truth[:, -1].max()
                    )

                self.optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if self.type == "deterministic":
                        loss = self.deterministic_loss(model_input, ground_truth)
                    else:
                        loss = self.probabilistic_loss(
                            model_input,
                            ground_truth,
                            no_reward=no_reward,
                            pre_fn=self.pre_fn,
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

                avg_train_loss += loss.item()
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(self.optim)
                scaler.update()

                if max_n_train_batches != -1 and max_n_train_batches <= batches_trained:
                    stop_training = True
                    break

                batches_trained += 1

            if debug:
                print("")
                print(f"Train loss: {avg_train_loss / n_train_batches}")

            avg_val_losses = torch.zeros((self.n_networks))

            for i_network in range(self.n_networks):
                for _ in range(n_val_batches):
                    (
                        model_input,
                        ground_truth,
                    ) = get_model_input_and_ground_truth_from_batch(
                        data.sample_val_batch(batch_size, val_split), device
                    )

                    if self.type == "deterministic":
                        avg_val_losses[i_network] += self.deterministic_loss(
                            model_input, ground_truth, i_network
                        ).item()
                    else:
                        avg_val_losses[i_network] += self.probabilistic_loss(
                            model_input,
                            ground_truth,
                            i_network,
                            only_mse=True,
                            no_reward=no_reward,
                            pre_fn=self.pre_fn,
                        ).item()

                avg_val_losses[i_network] /= n_val_batches

            avg_val_loss = avg_val_losses.mean()

            if avg_val_loss < min_val_loss:
                n_bad_val_losses = 0
                min_val_loss = avg_val_loss
            else:
                n_bad_val_losses += 1

            if stop_training:
                break

            if tuning:
                tune.report(val_loss=float(avg_val_loss.item()))
                with tune.checkpoint_dir(step=epoch) as tune_checkpoint_dir:
                    path = os.path.join(tune_checkpoint_dir, "checkpoint")
                    torch.save(
                        {
                            "step": epoch,
                            "model_state_dict": self.state_dict(),
                            "optim_state_dict": self.optim.state_dict(),
                            "val_loss": float(avg_val_loss.item()),
                        },
                        path,
                    )

            epoch += 1

            print(
                f"Train batches: {batches_trained} "
                f"Patience: {n_bad_val_losses}/{patience} "
                f"Val losses: {avg_val_losses.tolist()}",
                end="\r",
            )

            if debug:
                print("")

        self.has_been_trained_at_least_once = True
        return avg_val_losses, batches_trained

    def check_prediction_arguments(self, mode, pessimism):
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown mode: {mode}")

        if (
            pessimism != 0
            and mode in (ALEATORIC_PENALTY, ALEATORIC_PARTITIONING)
            and self.type == "deterministic"
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
            torch.Tensor: _description_
        """
        predicted_output, _, _, _, _, _ = self(model_input)

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
        no_reward: bool = False,
        only_uncertainty: bool = False,
        pre_fn: Callable = None,
        debug: bool = False,
    ):
        _, mean, logvar, max_logvar, min_logvar, uncertainty = self(model_input)

        if pre_fn is not None:
            mean[:, :, :-1] = pre_fn(mean[:, :, :-1], detach=False)
            ground_truth_output[:, :-1] = pre_fn(ground_truth_output[:, :-1])

        if no_reward:
            model_input = model_input[:, :-1]
            ground_truth_output = ground_truth_output[:, :-1]
            mean = mean[:, :, :-1]
            logvar = logvar[:, :, :-1]
            max_logvar = max_logvar[:, :-1]
            min_logvar = min_logvar[:, :-1]

        if i_network > -1:
            mean = mean[i_network]
            logvar = logvar[i_network]
            max_logvar = max_logvar[i_network]
            min_logvar = min_logvar[i_network]

        inv_var = torch.exp(-logvar)

        if only_mse:
            return torch.square(mean - ground_truth_output).mean()
        if only_uncertainty:
            return uncertainty.mean()

        mse_loss = torch.square(mean - ground_truth_output)
        mse_inv_var_loss = (mse_loss * inv_var).mean()
        var_loss = logvar.mean()
        var_bound_loss = 0.01 * max_logvar.mean() - 0.01 * min_logvar.mean()
        uncertainty_loss = uncertainty.mean()

        if debug:
            print(
                f"LR: {self.optim.param_groups[0]['lr']:.5f}, "
                f"State MSE: {mse_loss[:, :, :-1].mean().item():.5f}, "
                f"Rew MSE: {mse_loss[:, :, -1].mean().item():.5f}, "
                f"MSE + INV VAR: {mse_inv_var_loss.item():.5f} "
                f"VAR: {var_loss.item():.5f}, "
                f"BOUNDS: {var_bound_loss.item():.5f}, "
                f"MAX LOGVAR: {max_logvar.mean().item():.5f}, "
                f"MIN LOGVAR: {min_logvar.mean().item():.5f}, ",
                f"UNCERTAINTY: {uncertainty_loss.item():.5f}",
                end="\r",
            )

        return mse_inv_var_loss + var_loss + var_bound_loss + uncertainty_loss


def get_model_input_and_ground_truth_from_batch(
    batch: dict[str, torch.Tensor], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the model inputs and ground truth outputs from a batch.

    Args:
        batch (dict[str, torch.Tensor]): A batch sampled from a
            :py:class:`.ReplayBuffer`.
        device (str): The device to push the tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The model input and ground truth output from
            a batch.
    """
    model_input = torch.cat((batch["obs"], batch["act"]), dim=1)
    ground_truth = torch.cat((batch["obs2"], batch["rew"].unsqueeze(1)), dim=1)

    model_input = model_input.to(device)
    ground_truth = ground_truth.to(device)

    return model_input, ground_truth
