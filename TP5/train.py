import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader, test_loader
from text_utils import tokenize, untokenize, word2idx, idx2word, pretrained_embeddings
from model import ImageCaptioningModel, resnet_pretrained


# --- Initialisation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

model = ImageCaptioningModel(
    resnet_pretrained,
    pretrained_embeddings,
    lstm_hidden_dim=256
).to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(ignore_index=0)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)

os.makedirs("runs", exist_ok=True)
writer = SummaryWriter("runs/captioning")

num_epochs = 15
max_length = 20


# --- Préparation des légendes ---
def prepare_captions(captions):
    # Flickr30k peut fournir plusieurs légendes par image.
    # On utilise la première légende de chaque image.
    if len(captions) > 0 and isinstance(captions[0], str):
        sentences = captions
    else:
        sentences = captions[0]

    sequences = []
    for sentence in sentences:
        tokens = tokenize(sentence, word2idx, max_length=max_length)
        tokens = tokens[:max_length]
        tokens += [0] * (max_length - len(tokens))
        sequences.append(tokens)

    return torch.tensor(sequences, dtype=torch.long)


# --- Entraînement ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, (images, captions) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device)
        caps = prepare_captions(captions).to(device)

        outputs = model(images, caps[:, :-1])

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            caps[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            writer.add_scalar(
                "Loss/Train_Step",
                total_loss / 10,
                epoch * len(train_loader) + i
            )
            total_loss = 0.0

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    num_val_batches = 0
    pred_text = ""
    true_text = ""

    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)
            caps = prepare_captions(captions).to(device)

            outputs = model(images, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1)
            )

            val_loss += loss.item()
            num_val_batches += 1

            # Conserver une prédiction pour l'afficher
            predictions = outputs[0].argmax(dim=-1)

            pred_text = " ".join(
                untokenize(predictions.cpu().tolist(), idx2word)
            )
            true_text = " ".join(
                untokenize(caps[0, 1:].cpu().tolist(), idx2word)
            )

    avg_val_loss = val_loss / max(num_val_batches, 1)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Validation loss : {avg_val_loss:.4f}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

    scheduler.step()
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

writer.close()
