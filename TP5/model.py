
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from text_utils import pretrained_embeddings

# 1. Chargement de ResNet50 pré-entraîné
resnet_pretrained = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)


# 2. Suppression des deux dernières couches (AdaptiveAvgPool2d et Linear)
# On veut conserver les dimensions spatiales pour l'attention (pas un vecteur 1D global)
cnn_features = nn.Sequential(
    *list(resnet_pretrained.children())[:-2]
)


# 3. Gel des poids (Transfer Learning)
for param in cnn_features.parameters():
    param.requires_grad = False





class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()

        # Calcul d'un score d'attention pour chaque région spatiale
        self.attention_layer = nn.Conv2d(
            feature_dim + hidden_dim,
            1,
            kernel_size=1
        )

    def forward(self, features, hidden_state):
        # features : (batch_size, feature_dim, H, W)
        # hidden_state : (batch_size, hidden_dim)

        # Étaler l'état caché sur toute la grille spatiale
        hidden_state_expanded = (
            hidden_state.unsqueeze(2)
            .unsqueeze(3)
            .repeat(
                1, 1,
                features.size(2),
                features.size(3)
            )
        )

        # Concaténation des caractéristiques de l'image et de l'état du RNN (sur la dimension des canaux)
        concat_features = torch.cat(
            (features, hidden_state_expanded),
            dim=1
        )

        # Calcul des scores (batch_size, 1, 7, 7) -> on retire la dimension de taille 1
        attention_scores = self.attention_layer(
            concat_features
        ).squeeze(1)

        # Application du Softmax sur les 49 régions (7x7)
        attention_weights = F.softmax(
            attention_scores.view(attention_scores.size(0), -1),
            dim=1
        )
        attention_weights = attention_weights.view(
            attention_scores.size(0),
            1,
            attention_scores.size(1),
            attention_scores.size(2)
        )

	# Calcul du vecteur de contexte par somme pondérée sur les dimensions H et W (dim 2 et 3)
        context_vector = torch.sum(
            torch.sum(features * attention_weights, dim=3),
            dim=2
        )

        return context_vector, attention_weights




class LSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        attention_module,
        attention_size
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Entrée du LSTM = mot + état caché précédent + contexte visuel
        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size)
        self.lin_f = nn.Linear(total_dim, hidden_size)
        self.lin_c = nn.Linear(total_dim, hidden_size)
        self.lin_o = nn.Linear(total_dim, hidden_size)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, features, init_states=None):
        # x : (batch_size, sequence_size, input_size)
        bs, seq_sz, _ = x.size()

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(
                bs, self.hidden_size, device=x.device
            )
            c_t = torch.zeros(
                bs, self.hidden_size, device=x.device
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # Calcul du vecteur d'attention contextuel
            att_t, _ = self.attention_module(features, h_t)

            # Concaténation de l'attention, du mot actuel, et de l'état caché précédent
            concat = torch.cat((att_t, x_t, h_t), dim=1)

            # Equations du LSTM
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # Mise à jour de la mémoire et de l'état caché
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)



class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim):
        super(ImageCaptioningModel, self).__init__()
        # Backbone ResNet : extraction des caractéristiques visuelles
        self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
        # Geler les paramètres du CNN pré-entraîné
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.attention = AttentionModule(
            feature_dim=2048,
            hidden_dim=lstm_hidden_dim
        )
        self.embeddings = embedding_model
        vocab_size = embedding_model.weight.shape[0]
        emb_dim = embedding_model.weight.shape[1]

        self.lstm = LSTMWithAttention(
            emb_dim,
            lstm_hidden_dim,
            self.attention,
            2048
        )
        # Prédiction du mot suivant
        self.fc_out = nn.Linear(lstm_hidden_dim, vocab_size)
    def forward(self, images, captions):
        cnn_features = self.cnn(images)
        emb = self.embeddings(captions)

        lstm_outputs, (hidden_state, _) = self.lstm(
            emb, cnn_features
        )

        output = self.fc_out(lstm_outputs)
        return output




# 4. Test de l'extracteur de caractéristiques
if __name__ == "__main__":
    model = ImageCaptioningModel(
        resnet_pretrained,
        pretrained_embeddings,
        lstm_hidden_dim=256
    )

    model.eval()

    images = torch.randn(2, 3, 224, 224)
    captions = torch.randint(
        0, pretrained_embeddings.num_embeddings, (2, 5)
    )

    with torch.no_grad():
        outputs = model(images, captions)

    print("Shape des entrées texte :", captions.shape)
    print("Shape des sorties :", outputs.shape)
    print("Taille du vocabulaire :", pretrained_embeddings.num_embeddings)
    cnn_features.eval()

    images = torch.randn(32, 3, 224, 224)

    with torch.no_grad():
        features = cnn_features(images)

    print("Shape des images :", images.shape)
    print("Shape des caractéristiques :", features.shape)
    print(
        "Tous les paramètres sont gelés :",
        all(not p.requires_grad for p in cnn_features.parameters())
    )


    
    # Test du module d'attention
    batch_size = 2
    hidden_size = 256
    input_size = 300
    feature_dim = 2048

    attention = AttentionModule(
        feature_dim=feature_dim,
        hidden_dim=hidden_size
    )

    test_features = torch.randn(batch_size, feature_dim, 7, 7)
    test_hidden = torch.randn(batch_size, hidden_size)

    context, weights = attention(test_features, test_hidden)

    print("Shape du contexte :", context.shape)
    print("Shape des poids d'attention :", weights.shape)
    print("Somme des poids :", weights.sum(dim=(1, 2, 3)))

    # Test du LSTM personnalisé
    lstm = LSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        attention_module=attention,
        attention_size=feature_dim
    )

    test_sequence = torch.randn(batch_size, 5, input_size)

    hidden_seq, (h_final, c_final) = lstm(
        test_sequence,
        test_features
    )

    print("Shape de la séquence LSTM :", hidden_seq.shape)
    print("Shape de l'état caché final :", h_final.shape)
    print("Shape de l'état mémoire final :", c_final.shape)





