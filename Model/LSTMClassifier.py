import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support


# Basic LSTM for next-activity prediction
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_classes=10, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def forward(self, input_ids, lengths=None):
        # input_ids: [B, T] (left-padded)
        embedded = self.embedding(input_ids)      # [B, T, E]
        lstm_out, _ = self.lstm(embedded)         # [B, T, H]
        features = lstm_out[:, -1, :]             # last step corresponds to last real token (because left padding)
        logits = self.classifier(features)        # [B, C]
        return logits

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "vocab_size": int(self.embedding.num_embeddings),
            "num_classes": int(self.classifier.out_features),
            "embedding_dim": int(self.embedding.embedding_dim),
            "hidden_dim": int(self.lstm.hidden_size),
            "padding_idx": int(self.embedding.padding_idx) if self.embedding.padding_idx is not None else 0
        }, path)

    def expand_vocab(self, new_vocab_size: int):
        if new_vocab_size <= self.vocab_size:
            return

        old_emb: nn.Embedding = self.embedding
        old_num, dim = old_emb.weight.shape

        new_emb = nn.Embedding(new_vocab_size, dim, padding_idx=old_emb.padding_idx).to(old_emb.weight.device)

        # copy old weights
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old_emb.weight)
            # new rows remain randomly initialized (default init)

        self.embedding = new_emb
        self.vocab_size = new_vocab_size

    def expand_num_classes(self, new_num_classes: int):
        if new_num_classes <= self.num_classes:
            return

        old_fc: nn.Linear = self.classifier
        in_dim = old_fc.in_features
        old_out = old_fc.out_features

        new_fc = nn.Linear(in_dim, new_num_classes).to(old_fc.weight.device)

        with torch.no_grad():
            new_fc.weight[:old_out].copy_(old_fc.weight)
            new_fc.bias[:old_out].copy_(old_fc.bias)
            # new rows/bias are randomly initialized

        self.classifier = new_fc
        self.num_classes = new_num_classes


def _labels_to_index(labels: torch.Tensor) -> torch.Tensor:
    # LogsDataLoader.encode_labels returns one-hot [B, C] (dtype long) :contentReference[oaicite:4]{index=4}
    if labels.ndim == 2:
        return labels.argmax(dim=1)
    return labels



# Compute weighted Precision / Recall / F1 using sklearn.
# Unknown label (-1) is treated as wrong.
# preds: predictions / gts: ground-truth
def compute_prf1_weighted_sklearn(preds: torch.Tensor, gts: torch.Tensor):

    if len(preds) == 0 or len(gts) == 0:
        return 0.0, 0.0, 0.0

    preds_np = preds.detach().cpu().numpy()
    gts_np = gts.detach().cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        gts_np,
        preds_np,
        average="weighted",
        zero_division=0
    )

    return precision, recall, f1


@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn=None):
    device = next(model.parameters()).device
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, lengths in val_loader:
        inputs = inputs.to(device)
        labels = _labels_to_index(labels).to(device)

        logits = model(inputs)  # lengths not needed for left-padded baseline
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(model, dataloader, epochs=100, lr=0.002, patience=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = dataloader.dataset
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=dataloader.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=dataloader.batch_size, shuffle=False, num_workers=0)

    model.to(device)
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    stats = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for inputs, labels, lengths in train_loader:
            inputs = inputs.to(device)
            labels = _labels_to_index(labels).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running += loss.item() * labels.size(0)
            n += labels.size(0)

        train_loss = running / max(n, 1)
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn)

        stats["train_loss"].append(train_loss)
        stats["val_loss"].append(val_loss)
        stats["val_acc"].append(val_acc)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if bad >= patience:
            print(f"Early stopping. Best_val_loss={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, stats


@torch.no_grad()
def predict_model(model, test_dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, gts = [], []
    for inputs, labels, lengths in test_dataloader:
        inputs = inputs.to(device)
        labels = _labels_to_index(labels).to(device)

        logits = model(inputs)
        pred = logits.argmax(dim=1)

        preds.append(pred.cpu())
        gts.append(labels.cpu())

    preds = torch.cat(preds) if preds else torch.tensor([])
    gts = torch.cat(gts) if gts else torch.tensor([])

    correct = (preds == gts).sum().item()
    total = len(gts)
    acc = correct / total if total > 0 else 0.0

    return acc, preds, gts
