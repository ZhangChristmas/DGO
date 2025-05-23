# ultimate_morph_generator/dgo_oracle/training_engine.py
from random import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict, Any, Union  # Added Union
import numpy as np
import os
import copy  # For deepcopying model for EWC/SI

from ..config import get_config, DGOOracleConfig, DGOTrainingConfig, DataManagementConfig
from ..utilities.logging_config import setup_logging
from ..utilities.type_definitions import CvImage, Label, ImagePath
from ..data_management.dataset_utils import create_dgo_dataloader  # To create dataloaders
from .dgo_model_handler import DGOModelHandler  # To access the model

logger = setup_logging()


# --- Continual Learning Regularizers ---
class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) regularizer."""

    def __init__(self, model: nn.Module, fisher_matrices: Dict[str, torch.Tensor],
                 opt_params: Dict[str, torch.Tensor], ewc_lambda: float):
        self.model = model
        self.fisher_matrices = fisher_matrices  # Parameter name -> Fisher information matrix diagonal
        self.opt_params = opt_params  # Parameter name -> Optimal parameters from previous task
        self.ewc_lambda = ewc_lambda

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        if not self.fisher_matrices or not self.opt_params:  # No prior task data
            return loss

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_matrices and name in self.opt_params:
                fisher = self.fisher_matrices[name]
                opt_param = self.opt_params[name]
                loss += (fisher * (param - opt_param).pow(2)).sum()
        return self.ewc_lambda * loss / 2.0  # Division by 2 is common


class SIRegularizer:  # <<<--- RENAMED FROM SI সঞ্চয়কারী
    """Synaptic Intelligence (SI) regularizer."""

    def __init__(self, model: nn.Module, prev_params: Dict[str, torch.Tensor],
                 param_importances: Dict[str, torch.Tensor], si_lambda: float,
                 epsilon: float = 1e-4):  # epsilon for numerical stability
        self.model = model
        self.prev_params = prev_params  # Parameters from the end of the last task
        self.param_importances = param_importances  # Omega values (per-parameter importance)
        self.si_lambda = si_lambda
        self.epsilon = epsilon  # For damping term in denominator

        self.initial_params_current_task: Dict[str, torch.Tensor] = {}
        self.path_integrals_current_task: Dict[str, torch.Tensor] = {}  # W_k in the SI paper

        self._store_initial_task_state()

    def _store_initial_task_state(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_params_current_task[name] = param.data.clone()
                self.path_integrals_current_task[name] = torch.zeros_like(param.data)

    def update_path_integrals_per_step(self, param_name: str, grad_data: torch.Tensor,
                                       param_data_before_step: torch.Tensor, param_data_after_step: torch.Tensor):
        """
        Updates the path integral for a specific parameter after one optimizer step.
        W_k += -g_i * (theta_i_after - theta_i_before)
        This method should be called by the training loop for each parameter after optimizer.step().
        """
        if param_name not in self.path_integrals_current_task:
            # This can happen if new layers are added or if some params are frozen/unfrozen
            # For simplicity, we'll only update for params present at task start
            # logger.debug(f"SI: Path integral update skipped for param '{param_name}' not in initial_params_current_task.")
            return

        delta_theta_step = param_data_after_step - param_data_before_step
        # grad_data is g_i (gradient that led to this step)
        self.path_integrals_current_task[param_name].add_(-grad_data * delta_theta_step)

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        if not self.param_importances or not self.prev_params:  # No omega or theta* from previous task
            return loss

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_importances and name in self.prev_params:
                importance = self.param_importances[name]  # Omega_k_prev_task
                prev_optimal_param = self.prev_params[name]  # Theta*_k_prev_task
                loss += (importance * (param - prev_optimal_param).pow(2)).sum()
        return self.si_lambda * loss

    def compute_new_importances(self) -> Dict[str, torch.Tensor]:
        """
        Computes new parameter importances (Omega) at the end of training on the current task.
        This uses the path integrals accumulated during the current task and the parameter
        values at the start (theta_prev_task_optimal which is self.prev_params) and end of current task.
        """
        new_importances_for_current_task_delta: Dict[str, torch.Tensor] = {}
        current_optimal_params = {name: p.data.clone() for name, p in self.model.named_parameters() if p.requires_grad}

        for name, param_after_task in current_optimal_params.items():
            if name in self.prev_params and name in self.path_integrals_current_task:
                param_before_task = self.prev_params[name]  # This was theta*_t-1
                path_integral_val = self.path_integrals_current_task[name]  # This is W_k for task t

                delta_theta_task = param_after_task - param_before_task  # (theta*_t - theta*_t-1)

                # Omega_delta_k_t = W_k_t / ( (theta*_t - theta*_t-1)^2 + epsilon )
                # This is the *change* in importance contributed by the current task.
                importance_delta_task = path_integral_val / (delta_theta_task.pow(2) + self.epsilon)

                # Ensure non-negativity if path integrals can be negative (though they shouldn't if loss decreases)
                importance_delta_task.clamp_(min=0)
                new_importances_for_current_task_delta[name] = importance_delta_task

        # Total importance is accumulated: Omega_t = Omega_t-1 + Omega_delta_t
        accumulated_new_importances = self.param_importances.copy()  # Start with Omega_t-1
        for name, delta_omega in new_importances_for_current_task_delta.items():
            if name in accumulated_new_importances:
                accumulated_new_importances[name] += delta_omega
            else:  # Parameter might be new
                accumulated_new_importances[name] = delta_omega

        # Reset path integrals for the *next* task's accumulation
        self._store_initial_task_state()  # Re-call to reset path_integrals and store new initial_params

        return accumulated_new_importances


class DGOTrainingEngine:
    """
    Handles the training and fine-tuning of the DGO model.
    """

    def __init__(self, model_handler: DGOModelHandler,
                 training_cfg: DGOTrainingConfig,
                 dgo_cfg: DGOOracleConfig,  # For CL strategy info
                 data_cfg: DataManagementConfig,  # For image properties
                 device: torch.device):
        self.model_handler = model_handler
        self.model = model_handler.model  # Direct access to the nn.Module
        self.train_cfg = training_cfg
        self.dgo_cfg = dgo_cfg
        self.data_cfg = data_cfg
        self.device = device

        # Continual Learning state (for EWC, SI, etc.)
        self.cl_regularizer: Optional[Union[EWCRegularizer, SIRegularizer]] = None  # <<<--- RENAMED
        self._fisher_matrices: Dict[str, torch.Tensor] = {}
        self._opt_params_prev_task: Dict[str, torch.Tensor] = {}  # This is theta*_t-1
        self._param_importances_si: Dict[str, torch.Tensor] = {}  # This is Omega_t-1

    def _get_optimizer(self) -> optim.Optimizer:  # Unchanged
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        lr = self.train_cfg.learning_rate
        wd = self.train_cfg.weight_decay
        if self.train_cfg.optimizer == "Adam":
            return optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        elif self.train_cfg.optimizer == "AdamW":
            return optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        elif self.train_cfg.optimizer == "SGD":
            return optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=wd)
        elif self.train_cfg.optimizer == "RMSprop":
            return optim.RMSprop(trainable_params, lr=lr, weight_decay=wd)
        else:
            logger.warning(
                f"Unsupported optimizer: {self.train_cfg.optimizer}. Defaulting to AdamW."); return optim.AdamW(
                trainable_params, lr=lr, weight_decay=wd)

    def _get_criterion(self) -> nn.Module:  # Unchanged
        if self.train_cfg.loss_function == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.train_cfg.loss_function == "NLLLoss":
            return nn.NLLLoss()
        else:
            logger.warning(
                f"Unsupported loss function: {self.train_cfg.loss_function}. Defaulting to CrossEntropyLoss."); return nn.CrossEntropyLoss()

    def _run_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, epoch_num: int, total_epochs: int,
                   is_finetuning: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            params_before_step = {}
            if isinstance(self.cl_regularizer,
                          SIRegularizer) and is_finetuning:  # Store params before step for SI W_k calc
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        params_before_step[name] = param.data.clone()

            optimizer.zero_grad()
            outputs = self.model(inputs)
            classification_loss = criterion(outputs, targets)

            cl_penalty = torch.tensor(0.0, device=self.device)
            if is_finetuning and self.cl_regularizer:
                cl_penalty = self.cl_regularizer.penalty()

            loss = classification_loss + cl_penalty
            loss.backward()

            # Store gradients for SI path integral update *before* optimizer.step() modifies them or params.
            grads_for_si_step = {}
            if isinstance(self.cl_regularizer, SIRegularizer) and is_finetuning:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grads_for_si_step[name] = param.grad.data.clone()

            optimizer.step()

            if isinstance(self.cl_regularizer, SIRegularizer) and is_finetuning:
                for name, param_after_step in self.model.named_parameters():
                    if param_after_step.requires_grad and name in params_before_step and name in grads_for_si_step:
                        self.cl_regularizer.update_path_integrals_per_step(
                            param_name=name,
                            grad_data=grads_for_si_step[name],
                            param_data_before_step=params_before_step[name],
                            param_data_after_step=param_after_step.data  # Current param data
                        )

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 50 == 0:
                logger.debug(f"Epoch {epoch_num + 1}/{total_epochs} Batch {batch_idx}/{len(dataloader)} "
                             f"CLS Loss: {classification_loss.item():.4f} "
                             f"CL Penalty: {cl_penalty.item():.4f} "
                             f"Total Batch Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        logger.info(
            f"Epoch {epoch_num + 1}/{total_epochs} finished. Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_epoch_loss

    def train_initial_model(self, train_data: List[Tuple[ImagePath, Label]],
                            val_data: Optional[List[Tuple[ImagePath, Label]]] = None):  # Unchanged
        logger.info("Starting initial DGO model training...")
        epochs = self.train_cfg.epochs_initial_training
        if epochs == 0: logger.info(
            "Initial training epochs set to 0. Skipping initial training."); self._update_cl_structures_after_task(
            train_data_for_fisher=train_data); return
        train_loader = create_dgo_dataloader(image_label_pairs=train_data, dgo_cfg=self.dgo_cfg, data_cfg=self.data_cfg,
                                             batch_size=self.train_cfg.batch_size, augment=True, shuffle=True,
                                             use_weighted_sampler=True)
        optimizer = self._get_optimizer();
        criterion = self._get_criterion()
        scheduler = None
        if self.train_cfg.scheduler_step_size and self.train_cfg.scheduler_gamma: scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.train_cfg.scheduler_step_size, gamma=self.train_cfg.scheduler_gamma)
        for epoch in range(epochs): self._run_epoch(train_loader, optimizer, criterion, epoch, epochs,
                                                    is_finetuning=False);_ = scheduler.step() if scheduler else None
        logger.info("Initial DGO model training complete.");
        self._update_cl_structures_after_task(train_data_for_fisher=train_data)

    def finetune_model(self,
                       finetune_data: List[Tuple[Union[CvImage, ImagePath], Label]]):  # Allow Union for finetune_data
        logger.info(f"Starting DGO model fine-tuning with {len(finetune_data)} samples...")
        epochs = self.train_cfg.epochs_finetuning
        if epochs == 0 or not finetune_data: logger.info("Fine-tuning epochs set to 0 or no data. Skipping."); return

        self._setup_cl_regularizer_for_finetuning()  # This will init SI's path_integrals if SI is used

        # Adapt finetune_data if it contains CvImage to ImagePath for create_dgo_dataloader
        processed_finetune_data: List[Tuple[ImagePath, Label]] = []
        temp_dir_finetune: Optional[str] = None
        if any(isinstance(item[0], np.ndarray) for item in finetune_data):  # Check if any CvImage present
            temp_dir_finetune = "./temp_finetune_cv_imgs_for_dl"
            os.makedirs(temp_dir_finetune, exist_ok=True)
            for i, (img_data, label) in enumerate(finetune_data):
                if isinstance(img_data, np.ndarray):  # CvImage
                    path_str = os.path.join(temp_dir_finetune, f"ft_img_{i}.png")
                    cv2.imwrite(path_str, img_data)
                    processed_finetune_data.append((ImagePath(path_str), label))
                else:  # Already ImagePath
                    processed_finetune_data.append((img_data, label))
        else:  # All are ImagePath already
            processed_finetune_data = finetune_data  # type: ignore

        finetune_loader = create_dgo_dataloader(image_label_pairs=processed_finetune_data, dgo_cfg=self.dgo_cfg,
                                                data_cfg=self.data_cfg, batch_size=self.train_cfg.batch_size,
                                                augment=False, shuffle=True)

        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = max(original_lr * 0.1, 1e-5)
        optimizer = self._get_optimizer()
        self.train_cfg.learning_rate = original_lr
        criterion = self._get_criterion()

        for epoch in range(epochs): self._run_epoch(finetune_loader, optimizer, criterion, epoch, epochs,
                                                    is_finetuning=True)

        logger.info("DGO model fine-tuning complete.")
        self._update_cl_structures_after_task(
            train_data_for_fisher=processed_finetune_data)  # Use the same data for Fisher/SI update

        if temp_dir_finetune:  # Cleanup temp images
            import shutil;
            shutil.rmtree(temp_dir_finetune, ignore_errors=True)

    def _setup_cl_regularizer_for_finetuning(self):  # <<<--- RENAMED SIRegularizer here
        strategy = self.dgo_cfg.continual_learning_strategy
        if strategy == "ewc" and self._fisher_matrices and self._opt_params_prev_task:
            self.cl_regularizer = EWCRegularizer(self.model, self._fisher_matrices, self._opt_params_prev_task,
                                                 self.train_cfg.ewc_lambda)
            logger.info("EWC regularizer set up for fine-tuning.")
        elif strategy == "si" and self._opt_params_prev_task:  # SI needs prev_params. param_importances can be empty for first finetune.
            self.cl_regularizer = SIRegularizer(self.model, self._opt_params_prev_task,
                                                self._param_importances_si,  # Pass current Omega (Omega_t-1)
                                                self.train_cfg.si_lambda)
            # SIRegularizer's __init__ calls _store_initial_task_state to prep for W_k accumulation.
            logger.info("SI regularizer set up for fine-tuning.")
        elif strategy != "none":
            logger.warning(
                f"CL strategy '{strategy}' requested, but prerequisites not met. No CL penalty will be applied.")
            self.cl_regularizer = None
        else:
            self.cl_regularizer = None

    def _update_cl_structures_after_task(self, train_data_for_fisher: List[
        Tuple[Union[ImagePath, CvImage], Label]]):  # <<<--- RENAMED SIRegularizer here
        logger.info(f"Updating CL structures after task using strategy: {self.dgo_cfg.continual_learning_strategy}")
        self.model.eval()

        # Store optimal parameters from THIS task (theta*_t) for the NEXT task.
        # This means self._opt_params_prev_task becomes theta*_t.
        self._opt_params_prev_task = {name: param.data.clone() for name, param in self.model.named_parameters() if
                                      param.requires_grad}

        if self.dgo_cfg.continual_learning_strategy == "ewc":
            self._compute_fisher_information(
                train_data_for_fisher)  # This computes F_t and updates self._fisher_matrices
            logger.info("Fisher matrices (F_t) updated for EWC.")

        elif self.dgo_cfg.continual_learning_strategy == "si":
            if isinstance(self.cl_regularizer, SIRegularizer):
                # Compute new Omega_t using W_k from task t, and (theta*_t - theta*_t-1)
                # The SIRegularizer instance holds W_k accumulated during the finetuning of task t.
                # Its self.prev_params was theta*_t-1. Its self.model.parameters() are now theta*_t.
                self._param_importances_si = self.cl_regularizer.compute_new_importances()  # This becomes Omega_t
                logger.info("Parameter importances (Omega_t) updated for SI.")
            else:
                logger.warning("SI CL strategy, but regularizer is not SIRegularizer. Cannot update SI importances.")
        # self.cl_regularizer itself is reset/recreated in _setup_cl_regularizer_for_finetuning for next task.

    def _compute_fisher_information(self, dataset_for_fisher: List[Tuple[Union[ImagePath, CvImage], Label]],
                                    max_samples: int = 500):  # Unchanged from previous, but ensure data handling
        if not dataset_for_fisher: logger.warning("No data for Fisher Info. Skipping."); return

        processed_fisher_data: List[Tuple[ImagePath, Label]] = []  # Similar data adaptation as in finetune_model
        temp_dir_fisher: Optional[str] = None
        if any(isinstance(item[0], np.ndarray) for item in dataset_for_fisher):
            temp_dir_fisher = "./temp_fisher_cv_imgs_for_dl"
            os.makedirs(temp_dir_fisher, exist_ok=True)
            # Sample *before* saving to disk if dataset_for_fisher is large
            if len(dataset_for_fisher) > max_samples: dataset_for_fisher = random.sample(dataset_for_fisher,
                                                                                         max_samples)

            for i, (img_data, label) in enumerate(dataset_for_fisher):
                if isinstance(img_data, np.ndarray):
                    path_str = os.path.join(temp_dir_fisher, f"fi_img_{i}.png");
                    cv2.imwrite(path_str, img_data)
                    processed_fisher_data.append((ImagePath(path_str), label))
                else:
                    processed_fisher_data.append((img_data, label))
        else:  # All ImagePath
            processed_fisher_data = dataset_for_fisher  # type: ignore
            if len(processed_fisher_data) > max_samples: processed_fisher_data = random.sample(processed_fisher_data,
                                                                                               max_samples)

        fisher_loader = create_dgo_dataloader(image_label_pairs=processed_fisher_data, dgo_cfg=self.dgo_cfg,
                                              data_cfg=self.data_cfg, batch_size=max(1, self.train_cfg.batch_size // 4),
                                              shuffle=False, augment=False)
        new_fisher_matrices = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters() if
                               param.requires_grad}
        self.model.eval();
        criterion = self._get_criterion()

        for inputs, targets in fisher_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device);
            self.model.zero_grad()
            outputs = self.model(inputs);
            log_probs = F.log_softmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1);
            sampled_labels = torch.multinomial(probs, 1).squeeze()
            for i in range(inputs.size(0)):
                self.model.zero_grad()
                loss_sample = F.nll_loss(log_probs[i].unsqueeze(0), sampled_labels[i].unsqueeze(0))
                loss_sample.backward(retain_graph=True if i < inputs.size(0) - 1 else False)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None: new_fisher_matrices[name] += param.grad.data.pow(
                        2)

        num_fisher_samples = len(processed_fisher_data)
        for name in new_fisher_matrices: new_fisher_matrices[name] /= num_fisher_samples  # Average over samples

        if self._fisher_matrices:
            for name in new_fisher_matrices:
                if name in self._fisher_matrices:
                    self._fisher_matrices[name] = (self._fisher_matrices[name] + new_fisher_matrices[name]) / 2.0
                else:
                    self._fisher_matrices[name] = new_fisher_matrices[name]
        else:
            self._fisher_matrices = new_fisher_matrices

        if temp_dir_fisher: import shutil; shutil.rmtree(temp_dir_fisher, ignore_errors=True)

# __main__ block from previous version for testing remains the same, just uses SIRegularizer.
# ... (omitted for brevity, it was for testing the engine)