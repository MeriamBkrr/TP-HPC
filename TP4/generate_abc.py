import os
import torch
import torch.nn.functional as F
from train_music import (
    MusicModel, vocab_size, char_to_index, index_to_char,
    device, generate_music
)

CHECKPOINT = "best_lstm_clip1.0.pth"

prefix = """X:1
T:Irish Reel
M:4/4
L:1/8
K:D
"""

# Recréer le modèle et charger les poids existants
model = MusicModel(vocab_size, embedding_dim=256, hidden_size=512,
                   rnn_type="lstm").to(device)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location=device, weights_only=True)
)
model.eval()

os.makedirs("TP4/generated", exist_ok=True)

for temperature in [0.2, 0.8, 2.0]:
    torch.manual_seed(42)
    abc = generate_music(
        model,
        prefix,
        output_length=300,
        temperature=temperature
    )

    path = f"TP4/generated/irish_reel_T{temperature}.abc"
    with open(path, "w", encoding="utf-8") as f:
        f.write(abc)

    print(f"Partition enregistrée : {path}")

print("Terminé ! Les trois partitions sont dans TP4/generated/")
