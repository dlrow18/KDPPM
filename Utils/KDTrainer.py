# KDTrainer.py
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.KDPrepare import encode_df_with_given_vocab

def set_adaptation_trainable(
    student: nn.Module,
    n_old_tokens: int,
):
    """
    Adaptation phase:
      - freeze LSTM
      - freeze old embedding rows
      - train new embedding rows + classifier

    Note:
      PyTorch embedding cannot set requires_grad per-row directly,
      so we let embedding weight require grad, then manually mask old rows' grads.
    """
    # 1) freeze LSTM
    for p in student.lstm.parameters():
        p.requires_grad = False

    # 2) embedding weight stays trainable globally, but old rows will be masked after backward
    student.embedding.weight.requires_grad = True

    # 3) classifier trainable
    for p in student.classifier.parameters():
        p.requires_grad = True



def set_distillation_trainable(student: nn.Module):
    """
    Distillation phase:
      unfreeze all student parameters
    """
    for p in student.parameters():
        p.requires_grad = True


def zero_old_embedding_grads(student: nn.Module, n_old_tokens: int):
    """
    During adaptation, only new embedding rows should update.
    Old rows' gradients are zeroed out after backward.
    """
    if student.embedding.weight.grad is not None:
        student.embedding.weight.grad[:n_old_tokens].zero_()


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

    set_adaptation_trainable(student, n_old_tokens=n_old_tokens)

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


def compute_old_class_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    n_old_classes: int,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KD only on the old-class subspace.

    student_logits: [B, C_old + C_new]
    teacher_logits: [B, C_old]
    """
    student_old = student_logits[:, :n_old_classes]

    log_p_student = F.log_softmax(student_old / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    loss_kd = loss_kd * (temperature ** 2)
    return loss_kd


def build_student_teacher_loaders(
    kd_batch,
    student_vocab_mapper,
    teacher_vocab_mapper,
    max_case_length: int,
    batch_size: int = 32,
):
    """
    student side:
      raw novel_df, expand tokens/labels already done outside or allowed here

    teacher side:
      aligned teacher_df, no vocab expansion, unknown labels allowed
    """
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

    teacher_loader = encode_df_with_given_vocab(
        df=kd_batch.teacher_df,
        vocab_mapper=teacher_vocab_mapper,
        max_case_length=max_case_length,
        batch_size=batch_size,
        shuffle=False,
        expand_tokens=False,
        expand_labels=False,
        allow_unknown_labels=True,
    )
    return student_loader, teacher_loader


def train_kd_epoch(
    student: nn.Module,
    teacher: nn.Module,
    student_loader,
    teacher_loader,
    n_old_classes: int,
    lambda_kd: float = 1.0,
    temperature: float = 2.0,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    One epoch on D_novel:
      L = CE + lambda_kd * KD(old-subspace)

    teacher is frozen
    """
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

    # zip is okay here because both loaders are built from the same df in same batch size
    for (s_inputs, s_labels, s_lengths), (t_inputs, _, t_lengths) in zip(student_loader, teacher_loader):
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
    device: Optional[torch.device] = None,
):
    """
    Two-phase minimal incremental update:

    Phase 1: Adaptation
      - CE only
      - freeze LSTM
      - freeze old embedding rows
      - train new embedding rows + classifier

    Phase 2: Distillation
      - CE + KD
      - all parameters trainable
    """
    # teacher-side fixed vocab mapper
    teacher_vocab_mapper = copy.deepcopy(loader.vocab_mapper)
    teacher_vocab_mapper.token_vocab = copy.deepcopy(old_token_vocab)
    teacher_vocab_mapper.label_vocab = copy.deepcopy(old_label_vocab)
    teacher_vocab_mapper.pad_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.pad_token]
    teacher_vocab_mapper.unk_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.unk_token]

    # student side uses already-expanded current mapper
    student_vocab_mapper = loader.vocab_mapper

    student_loader, teacher_loader = build_student_teacher_loaders(
        kd_batch=kd_batch,
        student_vocab_mapper=student_vocab_mapper,
        teacher_vocab_mapper=teacher_vocab_mapper,
        max_case_length=loader.max_case_length,
        batch_size=batch_size,
    )

    n_old_classes = len(old_label_vocab)
    n_old_tokens = len(old_token_vocab)

    history = {
        "adaptation": [],
        "distillation": [],
    }

    # ===== Phase 1: Adaptation =====
    for epoch in range(1, adaptation_epochs + 1):
        student, stats = train_adaptation_epoch(
            student=student,
            student_loader=student_loader,
            n_old_tokens=n_old_tokens,
            lr=adaptation_lr,
            device=device,
        )
        history["adaptation"].append(stats)
        print(
            f"[Adaptation] epoch={epoch:03d} "
            f"ce={stats['ce_loss']:.4f} "
            f"n={stats['n_samples']}"
        )

    # ===== Phase 2: Distillation =====
    if use_kd:
        set_distillation_trainable(student)

        for epoch in range(1, kd_epochs + 1):
            student, stats = train_kd_epoch(
                student=student,
                teacher=teacher,
                student_loader=student_loader,
                teacher_loader=teacher_loader,
                n_old_classes=n_old_classes,
                lambda_kd=lambda_kd,
                temperature=temperature,
                lr=kd_lr,
                device=device,
            )
            history["distillation"].append(stats)
            print(
                f"[KD Train] epoch={epoch:03d} "
                f"ce={stats['ce_loss']:.4f} "
                f"kd={stats['kd_loss']:.4f} "
                f"total={stats['total_loss']:.4f} "
                f"n={stats['n_samples']}"
            )
    else:
        print("[KD Train] skipped (use_kd=False)")

    return student, history


'''
def incremental_kd_update(
    kd_batch,
    teacher,
    student,
    loader,
    old_token_vocab: Dict[str, int],
    old_label_vocab: Dict[str, int],
    batch_size: int = 32,
    kd_epochs: int = 5,
    lambda_kd: float = 1.0,
    temperature: float = 2.0,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    Full minimal KD update on D_novel only.
    """
    # build teacher-side fixed vocab mapper
    teacher_vocab_mapper = copy.deepcopy(loader.vocab_mapper)
    teacher_vocab_mapper.token_vocab = copy.deepcopy(old_token_vocab)
    teacher_vocab_mapper.label_vocab = copy.deepcopy(old_label_vocab)
    teacher_vocab_mapper.pad_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.pad_token]
    teacher_vocab_mapper.unk_idx = teacher_vocab_mapper.token_vocab[teacher_vocab_mapper.unk_token]

    # student side uses the already-expanded current mapper
    student_vocab_mapper = loader.vocab_mapper

    student_loader, teacher_loader = build_student_teacher_loaders(
        kd_batch=kd_batch,
        student_vocab_mapper=student_vocab_mapper,
        teacher_vocab_mapper=teacher_vocab_mapper,
        max_case_length=loader.max_case_length,
        batch_size=batch_size,
    )

    n_old_classes = len(old_label_vocab)

    history = []
    for epoch in range(1, kd_epochs + 1):
        student, stats = train_kd_epoch(
            student=student,
            teacher=teacher,
            student_loader=student_loader,
            teacher_loader=teacher_loader,
            n_old_classes=n_old_classes,
            lambda_kd=lambda_kd,
            temperature=temperature,
            lr=lr,
            device=device,
        )
        history.append(stats)
        print(
            f"[KD Train] epoch={epoch:03d} "
            f"ce={stats['ce_loss']:.4f} "
            f"kd={stats['kd_loss']:.4f} "
            f"total={stats['total_loss']:.4f} "
            f"n={stats['n_samples']}"
        )

    return student, history
'''