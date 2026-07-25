import torch
from Model.LSTMClassifier import compute_prf1_weighted_sklearn

# function to compute unseen event ratio in a batch_df given the label_vocab snapshot
def compute_unseen_event_ratio_like_detector(batch_df, known_events):
    """
    Event-level unseen ratio using the same known-event reference as DriftDetector.

    Definition:
        unseen_event_ratio =
        (# unseen events in prefix tokens + next_act) /
        (# total events in prefix tokens + next_act)

    This is aligned with unseen_ratio, which marks a prefix as unseen if either:
        - its prefix contains unseen tokens, or
        - its next_act is unseen.
    """
    total_events = 0
    unseen_events = 0

    # Count events inside prefixes
    for prefix in batch_df["prefix"].astype(str).tolist():
        for act in prefix.split():
            total_events += 1
            if act not in known_events:
                unseen_events += 1

    # Count next_act as the target event of each prefix
    for act in batch_df["next_act"].astype(str).tolist():
        total_events += 1
        if act not in known_events:
            unseen_events += 1

    return unseen_events / total_events if total_events > 0 else 0.0



# function to compute weighted F1 score for a batch_df given the label_vocab snapshot
"""
        Event-level decomposition of initially unseen events.

        global_unseen:
            event not in initial_known_events

        local_unseen / current_unseen:
            event not in current_model_known_events

        learned_novel:
            event not in initial_known_events
            and event in current_model_known_events

        All counts are based on event occurrences in:
            prefix tokens + next_act
        """
def compute_novel_event_decomposition(
        batch_df,
        initial_known_events,
        current_model_known_events,
):
    total_events = 0

    global_unseen_count = 0
    local_unseen_count = 0
    learned_novel_count = 0

    events = []

    for prefix in batch_df["prefix"].astype(str).tolist():
        events.extend(prefix.split())

    events.extend(batch_df["next_act"].astype(str).tolist())

    for act in events:
        total_events += 1

        is_global_unseen = act not in initial_known_events
        is_local_unseen = act not in current_model_known_events
        is_learned_novel = is_global_unseen and (act in current_model_known_events)

        if is_global_unseen:
            global_unseen_count += 1

        if is_local_unseen:
            local_unseen_count += 1

        if is_learned_novel:
            learned_novel_count += 1

    if total_events == 0:
        return {
            "total_events": 0,

            "global_unseen_event_count": 0,
            "global_unseen_event_ratio": 0.0,

            "local_unseen_event_count": 0,
            "local_unseen_event_ratio": 0.0,

            "learned_novel_event_count": 0,
            "learned_novel_event_ratio": 0.0,
        }

    return {
        "total_events": total_events,

        "global_unseen_event_count": global_unseen_count,
        "global_unseen_event_ratio": global_unseen_count / total_events,

        "local_unseen_event_count": local_unseen_count,
        "local_unseen_event_ratio": local_unseen_count / total_events,

        "learned_novel_event_count": learned_novel_count,
        "learned_novel_event_ratio": learned_novel_count / total_events,
    }


"""
        Compute acc / weighted P/R/F1 on a subset.
"""
def subset_metrics(preds, gts, mask):
    mask = torch.tensor(mask, dtype=torch.bool)

    n = int(mask.sum().item())
    if n == 0:
        return {
            "n": 0,
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    sub_preds = preds[mask]
    sub_gts = gts[mask]

    acc = float((sub_preds == sub_gts).sum().item() / n)
    p, r, f1 = compute_prf1_weighted_sklearn(sub_preds, sub_gts)

    return {
        "n": n,
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def overall_subset_metrics(pred_list, gt_list):
    if len(gt_list) == 0:
        return {
            "n": 0,
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    preds_tensor = torch.stack(pred_list)
    gts_tensor = torch.stack(gt_list)

    n = len(gt_list)
    acc = float((preds_tensor == gts_tensor).sum().item() / n)
    p, r, f1 = compute_prf1_weighted_sklearn(preds_tensor, gts_tensor)

    return {
        "n": n,
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


"""
        Split window samples into evaluation groups.

        1. current_unseen_target:
           target is not known before this window update.

        2. learned_novel_target:
           target was once novel, but has already been learned.

        3. novel_context_old_target:
           prefix contains a learned novel event, but target is old/known.

        4. old_target:
           target is currently known.
        """
def build_unseen_event_eval_masks(batch_df, current_known_events, learned_novel_events):
    next_acts = batch_df["next_act"].astype(str).tolist()
    prefixes = batch_df["prefix"].astype(str).tolist()

    current_unseen_target_mask = [
        y not in current_known_events
        for y in next_acts
    ]

    old_target_mask = [
        y in current_known_events
        for y in next_acts
    ]

    learned_novel_target_mask = [
        (y in learned_novel_events) and (y in current_known_events)
        for y in next_acts
    ]

    prefix_has_learned_novel_mask = []
    for p in prefixes:
        toks = p.split()
        prefix_has_learned_novel_mask.append(
            any(tok in learned_novel_events for tok in toks)
        )

    novel_context_old_target_mask = [
        prefix_has_novel and old_target
        for prefix_has_novel, old_target in zip(
            prefix_has_learned_novel_mask,
            old_target_mask
        )
    ]

    return {
        "current_unseen_target": current_unseen_target_mask,
        "learned_novel_target": learned_novel_target_mask,
        "novel_context_old_target": novel_context_old_target_mask,
    }

def compute_uhs(
    learned_novel_target_recall,
    novel_context_old_target_acc,
    alpha=0.5,
):
    """
    Compute the Unseen-event Handling Score.

    Rules:
    - both components available:
        weighted combination
    - only learned-novel-target recall available:
        use recall
    - only novel-context-old-target accuracy available:
        use accuracy
    - neither available:
        return None
    """
    if (
        learned_novel_target_recall is not None
        and novel_context_old_target_acc is not None
    ):
        return (
            alpha * learned_novel_target_recall
            + (1 - alpha) * novel_context_old_target_acc
        )

    if learned_novel_target_recall is not None:
        return learned_novel_target_recall

    if novel_context_old_target_acc is not None:
        return novel_context_old_target_acc

    return None
