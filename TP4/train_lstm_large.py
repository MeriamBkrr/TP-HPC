import torch
from train_music import train_model, generate_music

print("\n=== Entraînement LSTM grande échelle ===", flush=True)

model = train_model(
    rnn_type="lstm",
    subset_size=50000,
    num_epochs=30,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=1024,
    max_grad_norm=1.0,
    patience=30,
)

prefix = """X:1
T:Irish Reel
M:4/4
L:1/8
K:D
"""

for temperature in [0.2, 0.8, 2.0]:
    abc = generate_music(
        model,
        prefix,
        output_length=300,
        temperature=temperature,
    )

    print(f"\n{'='*20} TEMPÉRATURE {temperature} {'='*20}")
    print(abc, flush=True)

    path = f"TP4/generated/irish_reel_large_T{temperature}.abc"
    with open(path, "w", encoding="utf-8") as f:
        f.write(abc)

    print(f"Fichier enregistré : {path}", flush=True)
