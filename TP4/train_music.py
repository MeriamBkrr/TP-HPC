
import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reproductibilité
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/mnt/ssd/irishman"


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])



# 1. Extraction déterministe des caractères uniques
all_text = "".join(
    song["abc notation"] for song in train_data + val_data
)

unique_chars = sorted(set(all_text))
vocab_size = len(unique_chars)

print(f"\nTaille du vocabulaire : {vocab_size} caractères uniques")

# 2. Mappings caractère <-> index
char_to_index = {
    char: idx for idx, char in enumerate(unique_chars)
}

index_to_char = {
    idx: char for idx, char in enumerate(unique_chars)
}


def vectorize_string(s):
    return [char_to_index[char] for char in s]


# 3. Statistiques de longueur et Padding
lengths = [len(song["abc notation"]) for song in train_data]

max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")


SEQ_LENGTH = 500


def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    else:
        return sequence[:max_length]



class MusicDataset(Dataset):
    def __init__(self, raw_data, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(
                pad_or_truncate(song["abc notation"], seq_length)
            )
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)

        # Cible : tous les caractères sauf le premier
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y



class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()

        self.rnn_type = rnn_type.lower()

        # 1. Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Couche récurrente
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, batch_first=True
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim, hidden_size, batch_first=True
            )
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        # 3. Couche de sortie
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        embeds = self.embedding(x)
        out, hidden_state = self.rnn(embeds, hidden_state)
        logits = self.fc(out)
        return logits, hidden_state


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )



# Vérifications
example = train_data[0]["abc notation"]
normalized = pad_or_truncate(example)
vectorized = vectorize_string(example)

print(f"\nLongueur de l'exemple original : {len(example)}")
print(f"Longueur après padding/troncature : {len(normalized)}")
print(f"Longueur de la version vectorisée : {len(vectorized)}")
print(f"Index du caractère 'A' : {char_to_index.get('A', 'absent')}")


# Vérification du MusicDataset
train_dataset = MusicDataset(
    train_data, seq_length=SEQ_LENGTH, subset_size=100
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

x, y = next(iter(train_loader))

print("\n--- Vérification MusicDataset ---")
print("Taille du dataset :", len(train_dataset))
print("Shape de x :", x.shape)
print("Shape de y :", y.shape)
print("Type de x :", x.dtype)
print("Type de y :", y.dtype)





# Vérification de MusicModel
model_rnn = MusicModel(vocab_size, rnn_type="rnn")
model_lstm = MusicModel(vocab_size, rnn_type="lstm")

print("\n--- Vérification MusicModel ---")
print("Paramètres RNN :", count_parameters(model_rnn))
print("Paramètres LSTM :", count_parameters(model_lstm))

logits, hidden = model_lstm(x)
print("Shape de x :", x.shape)
print("Shape des logits :", logits.shape)








# ============================================================
# EXERCICE 3 : Entrainement, gradients et Early Stopping
# ============================================================

@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _ = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * y.numel()
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    return total_loss / total_tokens, total_correct / total_tokens


def train_model(
    rnn_type="lstm",
    subset_size=10000,
    num_epochs=15,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=512,
    max_grad_norm=None,
    patience=5,
):
    # 1. Préparation des données
    train_dataset = MusicDataset(
        train_data, subset_size=subset_size
    )
    val_dataset = MusicDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Création du modèle et de l'optimiseur
    model = MusicModel(
        vocab_size,
        embedding_dim,
        hidden_size,
        rnn_type=rnn_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    clip_tag = (
        f"clip{max_grad_norm}"
        if max_grad_norm is not None
        else "noclip"
    )

    run_name = (
        f"{rnn_type}_sub{subset_size}_lr{lr}_"
        f"{clip_tag}_{time.strftime('%H%M%S')}"
    )

    writer = SummaryWriter(log_dir=f"runs_rnn/{run_name}")
    save_path = f"best_{rnn_type}_{clip_tag}.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    print(f"\nDébut de l'expérience : {run_name}")
    print(f"Nombre de paramètres : {count_parameters(model)}")

    # 3. Boucle d'entraînement
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0
        epoch_grad_norms = []

        for x, y in tqdm(
            train_loader,
            desc=f"[{run_name}] Epoch {epoch}/{num_epochs}",
            leave=False
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            # Calcul des gradients
            loss.backward()

            # Mesure de la norme et clipping éventuel
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf")
                )

            epoch_grad_norms.append(grad_norm.item())

            if global_step % 20 == 0:
                writer.add_scalar(
                    "GradNorm/step",
                    grad_norm.item(),
                    global_step
                )

            # Mise à jour des poids après le clipping
            optimizer.step()

            total_train_loss += loss.item() * y.numel()
            total_train_tokens += y.numel()
            global_step += 1

        avg_train_loss = total_train_loss / total_train_tokens
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

        # 4. Évaluation
        val_loss, val_acc = evaluate_model(
            model, val_loader, loss_fn, vocab_size
        )

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("GradNorm/Epoch_Avg", avg_grad_norm, epoch)
        writer.flush()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Avg GradNorm: {avg_grad_norm:.3f}"
        )

        # 5. Early Stopping : sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Meilleur modèle sauvegardé : {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early Stopping à l'époque {epoch}.")
            break

    writer.close()

    # Recharger les meilleurs poids obtenus sur la validation
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )

    print(f"Expérience terminée : {run_name}")
    print(f"Meilleure Val Loss : {best_val_loss:.4f}")

    return model




@torch.no_grad()
def generate_music(model, start_sequence, output_length=300, temperature=1.0):
    model.eval()
    generated_seq = start_sequence

    input_indices = [char_to_index[c] for c in start_sequence]
    input_tensor = torch.tensor(
        input_indices, dtype=torch.long, device=device
    ).unsqueeze(0)

    logits, hidden_state = model(input_tensor, hidden_state=None)

    for _ in range(output_length):
        last_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1).squeeze(0)
        next_index = torch.multinomial(probs, num_samples=1).item()
        next_char = index_to_char[next_index]

        generated_seq += next_char
        input_tensor = torch.tensor(
            [[next_index]], dtype=torch.long, device=device
        )
        logits, hidden_state = model(input_tensor, hidden_state=hidden_state)

    return generated_seq


# ============================================================
# LANCEMENT DES 3 EXPERIENCES DEMANDEES
# ============================================================


if __name__ == "__main__":

    # Run 1 : RNN sans Gradient Clipping
    model_rnn_no_clip = train_model(
        rnn_type="rnn",
        max_grad_norm=None,
        subset_size=10000,
        num_epochs=15
    )

    # Run 2 : RNN avec Gradient Clipping
    model_rnn_clip = train_model(
        rnn_type="rnn",
        max_grad_norm=1.0,
        subset_size=10000,
        num_epochs=15
    )

    # Run 3 : LSTM avec Gradient Clipping
    model_lstm_clip = train_model(
        rnn_type="lstm",
        max_grad_norm=1.0,
        subset_size=10000,
        num_epochs=15
    )

    print("\nLes trois expériences sont terminées.")
