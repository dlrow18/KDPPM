import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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

        # print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | bad={bad}/{patience}")
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if bad >= patience:
            print(f"Early stopping. Best val_loss={best_val:.4f}")
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
    acc = float((preds == gts).float().mean().item()) if len(preds) else 0.0

    return acc, preds, gts





'''
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_classes=10, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits: FloatTensor [B, num_classes]
        """
        embedded = self.embedding(input_ids)  # [B, T, E]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)

        # hidden: [num_layers, B, H] -> take last layer
        features = hidden[-1]               # [B, H]
        logits = self.classifier(features)  # [B, C]
        return logits

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "vocab_size": int(self.embedding.num_embeddings),
            "num_classes": int(self.classifier.out_features),
            "embedding_dim": int(self.embedding.embedding_dim),
            "hidden_dim": int(self.lstm.hidden_size),
        }, path)

def train_model(model, dataloader, epochs=100, lr=0.002, patience=10, device=None):
    """
        Train the model on the given dataset.

        Args:
            dataloader: DataLoader with training and validation batches.
            lr: Learning rate for the optimizer.
            epochs: Number of training epochs.
            device: 'cpu' or 'cuda' if GPU available.
        """

    # Split dataset into training and validation sets
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    valid_size = int(dataset_size * 0.2)
    train_size = dataset_size - valid_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create training and validation data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    )

    print(f"Split dataset: {train_size} training samples, {valid_size} validation samples")

    # Move model to device
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.NAdam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        momentum_decay=0.004
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    training_stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle one-hot encoded labels
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_stats['train_loss'].append(avg_train_loss)

        # Validation phase
        valid_loss, valid_acc = evaluate_model(model, valid_loader, loss_fn)
        training_stats['val_loss'].append(valid_loss)
        training_stats['val_acc'].append(valid_acc)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_val_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}")

    return model, training_stats


def evaluate_model(model, val_loader, loss_fn=None):
    """Evaluate the model on a dataset.

    Args:
        model: The model to be evaluated.
        dataloader: DataLoader with validation batches.
        criterion: Loss function.
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)
            loss = loss_fn(outputs, labels) if loss_fn else 0
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def predict_model(model, test_dataloader, device=None):
    """Predict the model on the given dataset.

    Args:
        model: The model to be predicted.
        test_dataloader: DataLoader with test batches.
        device: Device to use for computations.

    Returns:
        Tuple of (accuracy, predictions, ground_truth)
    """
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    predictions = []
    ground_truth = []

    try:
        with torch.no_grad():
            for inputs, labels, _ in test_dataloader:
                if labels.ndim == 2:
                    labels = labels.argmax(dim=1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
                ground_truth.append(labels)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Model vocabulary size: {len(model.embed.vocab)}")
        print(f"Model output classes: {model.classifier.out_features}")
        raise

    # Compute accuracy
    if not predictions or not ground_truth:
        return 0.0, torch.tensor([]), torch.tensor([])

    try:
        all_predictions = torch.cat(predictions)
        all_ground_truth = torch.cat(ground_truth)
        correct = (all_predictions == all_ground_truth).sum().item()
        total = all_ground_truth.size(0)
        accuracy = correct / total if total > 0 else 0.0

        return accuracy, all_predictions, all_ground_truth
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return 0.0, torch.tensor([]), torch.tensor([])
'''