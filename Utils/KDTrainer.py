from __future__ import annotations
import copy
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.KDPrepare import encode_df_with_given_vocab

# Trainability helpers

#     Set for Adaptation phase:
#       - freeze LSTM and old embedding rows
#       - train new embedding rows + classifier
def set_adaptation_trainable(
    student: nn.Module
):

    # 1) freeze LSTM
    for p in student.lstm.parameters():
        p.requires_grad = False

    # 2) embedding weight stays trainable globally, but old rows will be masked after backward
    student.embedding.weight.requires_grad = True

    # 3) classifier trainable
    for p in student.classifier.parameters():
        p.requires_grad = True

# Set all student parameters unfreeze for distillation phase and full finetuning phase.
def set_distillation_trainable(student: nn.Module):

    for p in student.parameters():
        p.requires_grad = True


# Set all student parameters unfreeze for full-parameter CE-only finetuning
def set_full_finetune_trainable(student: nn.Module):

    for p in student.parameters():
        p.requires_grad = True

# During adaptation, only new embedding rows should update.
#     Old rows' gradients are zeroed out after backward.
def zero_old_embedding_grads(student: nn.Module, n_old_tokens: int):

    if student.embedding.weight.grad is not None:
        student.embedding.weight.grad[:n_old_tokens].zero_()


# Loader builders

# Build student dataloaders for the adaptation phase.
def build_novel_student_loader(
    kd_batch,
    student_vocab_mapper,
    max_case_length: int,
    batch_size: int = 32,
):
    student_loader = encode_df_with_given_vocab(
        df=kd_batch.novel_df,
        vocab_mapper=student_vocab_mapper,
        max_case_length=max_case_length,
        batch_size=batch_size,
        shuffle=True,
        expand_tokens=False,
        expand_labels=False,
        allow_unknown_labels=False,
    )


    return student_loader

# Build student and teacher dataloaders for the distillation phase on D_stable.
def build_stable_loaders(
    kd_batch,
    student_vocab_mapper,
    teacher_vocab_mapper,
    max_case_length: int,
    batch_size: int = 32,
):

    if len(kd_batch.stable_df) == 0:
        return None, None

    stable_student_loader = encode_df_with_given_vocab(
        df=kd_batch.stable_df,
        vocab_mapper=student_vocab_mapper,
        max_case_length=max_case_length,
        batch_size=batch_size,
        shuffle=True,
        expand_tokens=False,
        expand_labels=False,
        allow_unknown_labels=False,
    )

    stable_teacher_loader = encode_df_with_given_vocab(
        df=kd_batch.stable_df,
        vocab_mapper=teacher_vocab_mapper,
        max_case_length=max_case_length,
        batch_size=batch_size,
        shuffle=False,
        expand_tokens=False,
        expand_labels=False,
        allow_unknown_labels=False,
    )
    return stable_student_loader, stable_teacher_loader

# Helper for adaptation phase

def make_train_val_loader(
    dataloader,
    val_ratio: float = 0.2,
    shuffle_train: bool = True,
):
    """Split a dataloader's dataset into train/val loaders.

    Returns:
        (train_loader, val_loader). val_loader is None when the dataset is too
        small to form a non-empty validation set.
    """
    dataset = dataloader.dataset
    dataset_size = len(dataset)

    if dataset_size <= 1 or val_ratio <= 0:
        train_loader = DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=shuffle_train,
            num_workers=getattr(dataloader, "num_workers", 0),
        )
        return train_loader, None

    val_size = int(dataset_size * val_ratio)
    if val_size <= 0:
        val_size = 1
    train_size = dataset_size - val_size

    if train_size <= 0:
        train_size = dataset_size - 1
        val_size = 1

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle_train,
        num_workers=getattr(dataloader, "num_workers", 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=getattr(dataloader, "num_workers", 0),
    )
    return train_loader, val_loader

# Helper for distillation phase
def compute_old_class_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    n_old_classes: int,
    temperature: float = 2.0,
) -> torch.Tensor:

    student_old = student_logits[:, :n_old_classes]

    log_p_student = F.log_softmax(student_old / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    loss_kd = loss_kd * (temperature ** 2)
    return loss_kd



# Epoch-level functions

def train_adaptation_epoch(
    student: nn.Module,
    student_loader,
    n_old_tokens: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    One epoch on D_novel:
      L = CE
    Update:
      - new embedding rows
      - classifier
    Freeze:
      - LSTM
      - old embedding rows (via grad masking)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    student.train()

    set_adaptation_trainable(student)

    optimizer = torch.optim.NAdam(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    total_ce = 0.0
    total_n = 0

    for s_inputs, s_labels, s_lengths in student_loader:
        s_inputs = s_inputs.to(device)
        s_labels = s_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        student_logits = student(s_inputs)
        loss_ce = ce_loss_fn(student_logits, s_labels)

        loss_ce.backward()

        # keep old embedding rows frozen
        zero_old_embedding_grads(student, n_old_tokens=n_old_tokens)

        optimizer.step()

        bs = s_labels.size(0)
        total_ce += loss_ce.item() * bs
        total_n += bs

    stats = {
        "ce_loss": total_ce / max(total_n, 1),
        "n_samples": total_n,
    }
    return student, stats


def train_full_ce_epoch(
    student: nn.Module,
    student_loader,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    One epoch on D_novel:
      L = CE
    Update:
      all parameters trainable
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    student.train()

    set_full_finetune_trainable(student)

    optimizer = torch.optim.NAdam(student.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_ce = 0.0
    total_n = 0

    for s_inputs, s_labels, s_lengths in student_loader:
        s_inputs = s_inputs.to(device)
        s_labels = s_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        student_logits = student(s_inputs)
        loss_ce = ce_loss_fn(student_logits, s_labels)

        loss_ce.backward()
        optimizer.step()

        bs = s_labels.size(0)
        total_ce += loss_ce.item() * bs
        total_n += bs

    stats = {
        "ce_loss": total_ce / max(total_n, 1),
        "n_samples": total_n,
    }
    return student, stats


def train_stable_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    stable_student_loader,
    stable_teacher_loader,
    n_old_classes: int,
    lambda_kd: float = 1.0,
    temperature: float = 2.0,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    One epoch on D_stable:
      L = CE + lambda_kd * KD(old-subspace)
    """
    if stable_student_loader is None or stable_teacher_loader is None:
        return student, {
            "ce_loss": 0.0,
            "kd_loss": 0.0,
            "total_loss": 0.0,
            "n_samples": 0,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    student.train()

    optimizer = torch.optim.NAdam(student.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    total_ce = 0.0
    total_kd = 0.0
    total_loss = 0.0
    total_n = 0

    for (s_inputs, s_labels, _), (t_inputs, _, _) in zip(stable_student_loader, stable_teacher_loader):
        s_inputs = s_inputs.to(device)
        s_labels = s_labels.to(device)
        t_inputs = t_inputs.to(device)

        optimizer.zero_grad(set_to_none=True)

        student_logits = student(s_inputs)
        with torch.no_grad():
            teacher_logits = teacher(t_inputs)

        loss_ce = ce_loss_fn(student_logits, s_labels)
        loss_kd = compute_old_class_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            n_old_classes=n_old_classes,
            temperature=temperature,
        )

        loss = loss_ce + lambda_kd * loss_kd
        loss.backward()
        optimizer.step()

        bs = s_labels.size(0)
        total_ce += loss_ce.item() * bs
        total_kd += loss_kd.item() * bs
        total_loss += loss.item() * bs
        total_n += bs

    stats = {
        "ce_loss": total_ce / max(total_n, 1),
        "kd_loss": total_kd / max(total_n, 1),
        "total_loss": total_loss / max(total_n, 1),
        "n_samples": total_n,
    }
    return student, stats


@torch.no_grad()
def evaluate_adaptation(
    student: nn.Module,
    val_loader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate adaptation phase on a validation split."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    student.eval()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_ce = 0.0
    total_n = 0
    total_correct = 0

    for s_inputs, s_labels, s_lengths in val_loader:
        s_inputs = s_inputs.to(device)
        s_labels = s_labels.to(device)

        student_logits = student(s_inputs)
        loss_ce = ce_loss_fn(student_logits, s_labels)

        bs = s_labels.size(0)
        total_ce += loss_ce.item() * bs
        total_n += bs
        total_correct += (student_logits.argmax(dim=1) == s_labels).sum().item()

    return {
        "val_ce_loss": total_ce / max(total_n, 1),
        "val_acc": total_correct / max(total_n, 1),
        "n_val_samples": total_n,
    }


# Phase-level functions
# Phase-level functions
def run_adaptation_phase(
    student: nn.Module,
    student_loader,
    n_old_tokens: int,
    adaptation_epochs: int = 3,
    adaptation_lr: float = 3e-3,
    full_finetune_ce_only: bool = False,
    adaptation_val_ratio: float = 0.2,
    adaptation_patience: int = 2,
    adaptation_min_delta: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    history = []

    # Print the adaptation mode
    if verbose:
        if full_finetune_ce_only:
            print(
                "[Adaptation] Mode: full-parameter CE fine-tuning on D_novel."
            )
        else:
            print(
                "[Adaptation] Mode: partial-parameter CE training on D_novel."
            )

    # Split D_novel into training and validation sets
    adaptation_train_loader, adaptation_val_loader = make_train_val_loader(
        student_loader,
        val_ratio=adaptation_val_ratio,
        shuffle_train=True,
    )

    # Case 1: no validation set can be created
    if adaptation_val_loader is None:

        if verbose:
            print(
                f"[Adaptation] Validation split skipped | "
                f"train={len(adaptation_train_loader.dataset)}."
            )
            print(
                f"[Adaptation] Started | "
                f"epochs={adaptation_epochs} | "
                f"validation=disabled."
            )

        for epoch in range(1, adaptation_epochs + 1):
            if full_finetune_ce_only:
                student, stats = train_full_ce_epoch(
                    student=student,
                    student_loader=adaptation_train_loader,
                    lr=adaptation_lr,
                    device=device,
                )
            else:
                student, stats = train_adaptation_epoch(
                    student=student,
                    student_loader=adaptation_train_loader,
                    n_old_tokens=n_old_tokens,
                    lr=adaptation_lr,
                    device=device,
                )

            history.append(stats)

        if verbose:
            final_loss = history[-1]["ce_loss"] if history else None

            if final_loss is not None:
                print(
                    f"[Adaptation] Completed | "
                    f"epochs={len(history)} | "
                    f"final_train_loss={final_loss:.4f}."
                )
            else:
                print("[Adaptation] Completed | no training epochs executed.")

        return student, history

    # Case 2: training and validation sets are available
    if verbose:
        print(
            f"[Adaptation] Data split: "
            f"train={len(adaptation_train_loader.dataset)} | "
            f"val={len(adaptation_val_loader.dataset)}."
        )
        print(
            f"[Adaptation] Started | "
            f"max_epochs={adaptation_epochs} | "
            f"patience={adaptation_patience}."
        )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0
    stopped_early = False

    for epoch in range(1, adaptation_epochs + 1):
        if full_finetune_ce_only:
            student, train_stats = train_full_ce_epoch(
                student=student,
                student_loader=adaptation_train_loader,
                lr=adaptation_lr,
                device=device,
            )
        else:
            student, train_stats = train_adaptation_epoch(
                student=student,
                student_loader=adaptation_train_loader,
                n_old_tokens=n_old_tokens,
                lr=adaptation_lr,
                device=device,
            )

        val_stats = evaluate_adaptation(
            student=student,
            val_loader=adaptation_val_loader,
            device=device,
        )

        stats = {**train_stats, **val_stats}
        history.append(stats)

        current_val_loss = val_stats["val_ce_loss"]
        improved = current_val_loss < (
            best_val_loss - adaptation_min_delta
        )

        if improved:
            best_val_loss = current_val_loss
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= adaptation_patience:
            stopped_early = True

            if verbose:
                print(
                    f"[Adaptation] Early stopping at epoch {epoch} | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_loss={best_val_loss:.4f}."
                )

            break

    # Restore the model with the lowest validation loss
    if best_state is not None:
        student.load_state_dict(best_state)
        student = student.to(device)

    if verbose:
        completed_epochs = len(history)

        if stopped_early:
            print(
                f"[Adaptation] Completed | "
                f"epochs={completed_epochs} | "
                f"restored_epoch={best_epoch} | "
                f"best_val_loss={best_val_loss:.4f}."
            )
        else:
            print(
                f"[Adaptation] Completed | "
            )

    return student, history

def run_stable_kd_phase(
    student: nn.Module,
    teacher: nn.Module,
    stable_student_loader,
    stable_teacher_loader,
    n_old_classes: int,
    kd_epochs: int = 5,
    lambda_kd: float = 1.0,
    temperature: float = 2.0,
    kd_lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    history = []

    set_distillation_trainable(student)

    stable_n = (
        len(stable_student_loader.dataset)
        if stable_student_loader is not None
        else 0
    )

    if verbose:
        print(
            f"[Distillation] Started | "
            f"samples={stable_n} | "
            f"epochs={kd_epochs} | "
        )

    for epoch in range(1, kd_epochs + 1):
        student, stable_stats = train_stable_kd_epoch(
            student=student,
            teacher=teacher,
            stable_student_loader=stable_student_loader,
            stable_teacher_loader=stable_teacher_loader,
            n_old_classes=n_old_classes,
            lambda_kd=lambda_kd,
            temperature=temperature,
            lr=kd_lr,
            device=device,
        )

        stats = {
            "stable_ce_loss": stable_stats["ce_loss"],
            "stable_kd_loss": stable_stats["kd_loss"],
            "stable_total_loss": stable_stats["total_loss"],
            "stable_n": stable_stats["n_samples"],
        }
        history.append(stats)

    if verbose:
        print(
            f"[Distillation] Completed "
        )

    return student, history


# Pipeline-level function
def incremental_kd_update(
    kd_batch,
    teacher,
    student,
    loader,
    old_token_vocab: Dict[str, int],
    old_label_vocab: Dict[str, int],
    batch_size: int = 32,
    adaptation_epochs: int = 3,
    kd_epochs: int = 5,
    lambda_kd: float = 1.0,
    temperature: float = 2.0,
    adaptation_lr: float = 3e-3,
    kd_lr: float = 1e-3,
    use_kd: bool = True,
    full_finetune_ce_only: bool = False,
    adaptation_val_ratio: float = 0.2,
    adaptation_patience: int = 2,
    adaptation_min_delta: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    # teacher-side fixed vocab mapper
    teacher_vocab_mapper = copy.deepcopy(loader.vocab_mapper)
    teacher_vocab_mapper.token_vocab = copy.deepcopy(old_token_vocab)
    teacher_vocab_mapper.label_vocab = copy.deepcopy(old_label_vocab)
    teacher_vocab_mapper.pad_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.pad_token]
    teacher_vocab_mapper.unk_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.unk_token]

    # student side uses already-expanded current mapper
    student_vocab_mapper = loader.vocab_mapper

    # 当前主流程只需要 D_novel 的 student_loader
    student_loader = build_novel_student_loader(
        kd_batch=kd_batch,
        student_vocab_mapper=student_vocab_mapper,
        max_case_length=loader.max_case_length,
        batch_size=batch_size,
    )

    stable_student_loader, stable_teacher_loader = build_stable_loaders(
        kd_batch=kd_batch,
        student_vocab_mapper=student_vocab_mapper,
        teacher_vocab_mapper=teacher_vocab_mapper,
        max_case_length=loader.max_case_length,
        batch_size=batch_size,
    )

    n_old_classes = len(old_label_vocab)
    n_old_tokens = len(old_token_vocab)

    student, adaptation_history = run_adaptation_phase(
        student=student,
        student_loader=student_loader,
        n_old_tokens=n_old_tokens,
        adaptation_epochs=adaptation_epochs,
        adaptation_lr=adaptation_lr,
        full_finetune_ce_only=full_finetune_ce_only,
        adaptation_val_ratio=adaptation_val_ratio,
        adaptation_patience=adaptation_patience,
        adaptation_min_delta=adaptation_min_delta,
        device=device,
        verbose=verbose,
    )

    if use_kd:
        student, distillation_history = run_stable_kd_phase(
            student=student,
            teacher=teacher,
            stable_student_loader=stable_student_loader,
            stable_teacher_loader=stable_teacher_loader,
            n_old_classes=n_old_classes,
            kd_epochs=kd_epochs,
            lambda_kd=lambda_kd,
            temperature=temperature,
            kd_lr=kd_lr,
            device=device,
            verbose=verbose,
        )
    else:
        if verbose:
            print("[Distillation] Skipped.")
        distillation_history = []

    history = {
        "adaptation": adaptation_history,
        "distillation": distillation_history,
    }

    return student, history
