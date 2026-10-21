"""Architectures : CNN-Plain, CNN-Res, backbones pré-entraînés, auto-encodeurs, VAE."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import config as C


# ==========================================================================
# Partie 1 : CNN entraînés de zéro
# ==========================================================================
def conv_bn_relu(c_in, c_out):
    """Conv 3×3, padding 1, stride 1 (taille spatiale conservée) + BN + ReLU.
    Pas de biais : le beta de la BatchNorm joue ce rôle."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class CNNPlain(nn.Module):
    """3 étages de 2 blocs Conv-BN-ReLU, MaxPool 2×2 après chaque étage, filtres 32 -> 64 -> 128.
    Tête : Flatten -> Dense -> BN -> ReLU -> Dropout -> Dense(n_classes)."""

    def __init__(self, n_classes=10, widths=(32, 64, 128), dense=32, dropout=0.3, in_size=128):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [conv_bn_relu(c, w), conv_bn_relu(w, w), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        side = in_size // 2 ** len(widths)
        # BatchNorm après la Dense : sans elle, 30 neurones sur 32 « mouraient » (ReLU toujours à 0)
        # car la couche reçoit 32 768 valeurs d'un coup (essai documenté dans le notebook 01).
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * side * side, dense), nn.BatchNorm1d(dense),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dense, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + raccourci, ReLU après l'addition (He et al., 2015).
    Raccourci = identité si les dimensions sont égales, sinon convolution 1×1 (+BN)."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CNNRes(nn.Module):
    """Même profondeur que CNN-Plain (6 convolutions 3×3) : 3 ResidualBlock suivis d'un MaxPool,
    puis Global Average Pooling + Dropout + Dense. Largeurs élargies (64 -> 128 -> 256) pour un
    budget de paramètres du même ordre que CNN-Plain, dont la tête Flatten+Dense est coûteuse."""

    def __init__(self, n_classes=10, widths=(64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, c = [], 3
        for w in widths:
            blocks += [ResidualBlock(c, w), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def conv3x3_names(model):
    """Noms des poids des 6 convolutions 3×3, dans l'ordre (pour la norme du gradient)."""
    return [n for n, p in model.named_parameters() if p.dim() == 4 and p.shape[-1] == 3]


# ==========================================================================
# Partie 2 : transfer learning
# ==========================================================================
class TLModel(nn.Module):
    """Backbone torchvision + nouvelle tête. Gère les 3 stratégies de fine-tuning.

    strategy : 'frozen'    -> backbone gelé, seule la tête apprend
               'last_third'-> dernier tiers des blocs dégelé
               'full'      -> tout dégelé (LR différencié dans l'optimiseur)
    Les BatchNorm des parties gelées restent en mode eval (statistiques ImageNet conservées)."""

    def __init__(self, arch=C.LIGHT_BACKBONE, n_classes=10, strategy="full", pretrained=C.PRETRAINED):
        super().__init__()
        self.arch, self.strategy = arch, strategy
        if arch == "mobilenet_v3_large":
            w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            net = torchvision.models.mobilenet_v3_large(weights=w)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
            self.blocks = list(net.features.children())
            self.head_modules = [net.classifier]
            self.cam_layer = net.features[-1]
        elif arch == "mobilenet_v3_small":
            w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.mobilenet_v3_small(weights=w)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
            self.blocks = list(net.features.children())
            self.head_modules = [net.classifier]
            self.cam_layer = net.features[-1]
        elif arch == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = torchvision.models.resnet50(weights=w)
            net.fc = nn.Linear(net.fc.in_features, n_classes)
            self.blocks = [nn.Sequential(net.conv1, net.bn1), net.layer1, net.layer2, net.layer3, net.layer4]
            self.head_modules = [net.fc]
            self.cam_layer = net.layer4[-1]
        else:
            raise ValueError(arch)
        self.net = net
        self.set_strategy(strategy)

    def head_params(self):
        return [p for m in self.head_modules for p in m.parameters()]

    def backbone_params(self):
        head = {id(p) for p in self.head_params()}
        return [p for p in self.net.parameters() if id(p) not in head and p.requires_grad]

    def set_strategy(self, strategy):
        self.strategy = strategy
        for p in self.net.parameters():
            p.requires_grad = strategy == "full"
        if strategy == "last_third":
            n = len(self.blocks)
            for b in self.blocks[n - max(1, round(n / 3)):]:
                for p in b.parameters():
                    p.requires_grad = True
        for p in self.head_params():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, x):
        return self.net(x)


def tl_optimizer(model: TLModel, lr_head=1e-3, lr_backbone=None):
    """Adam avec groupes de paramètres : LR de la tête et LR (plus petit) du backbone dégelé."""
    groups = [{"params": model.head_params(), "lr": lr_head}]
    bb = model.backbone_params()
    if bb:
        groups.append({"params": bb, "lr": lr_backbone if lr_backbone is not None else lr_head})
    return torch.optim.Adam(groups)


def build_classifier(meta: dict) -> nn.Module:
    """Reconstruit un classifieur à partir des métadonnées d'un checkpoint."""
    n = len(meta["classes"])
    if meta["arch"] == "cnn_plain":
        return CNNPlain(n, **meta.get("kwargs", {}))
    if meta["arch"] == "cnn_res":
        return CNNRes(n, **meta.get("kwargs", {}))
    return TLModel(meta["arch"], n, meta.get("strategy", "full"), pretrained=False)


def load_classifier(path, device=C.DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(ck["meta"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["meta"]


# ==========================================================================
# Parties 3 et 5 : auto-encodeurs
# ==========================================================================
class Encoder(nn.Module):
    """Conv stride 2 successives : 64 -> 32 -> 16 -> 8, canaux 3 -> 32 -> 64 -> 128."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),     # 64 -> 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),    # 32 -> 16
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16 -> 8
            nn.Flatten())
        self.out_dim = 128 * 8 * 8

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """Symétrique : Linear -> reshape 128×8×8 -> ConvTranspose 8 -> 16 -> 32 -> 64 -> Sigmoid."""

    def __init__(self, k):
        super().__init__()
        self.fc = nn.Linear(k, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 8 -> 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 16 -> 32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())                                 # 32 -> 64

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, 8, 8))


class ConvAE(nn.Module):
    """Auto-encodeur convolutif à goulot k (k << n = 64·64·3 = 12 288)."""

    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.encoder = Encoder()
        self.to_z = nn.Linear(self.encoder.out_dim, k)
        self.decoder = Decoder(k)

    def encode(self, x):
        return self.to_z(self.encoder(x))

    def forward(self, x):
        return self.decoder(self.encode(x))


class VAE(nn.Module):
    """Auto-encodeur variationnel : l'encodeur donne mu et log sigma², z = mu + sigma·eps."""

    def __init__(self, k=32):
        super().__init__()
        self.k = k
        self.encoder = Encoder()
        self.fc_mu = nn.Linear(self.encoder.out_dim, k)
        self.fc_logvar = nn.Linear(self.encoder.out_dim, k)
        self.decoder = Decoder(k)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device=C.DEVICE):
        return self.decoder(torch.randn(n, self.k, device=device))


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Perte = reconstruction (somme des erreurs au carré par image) + beta·KL, moyennée sur le batch."""
    rec = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()


def load_ae(path, device=C.DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    model = (VAE if meta.get("type") == "vae" else ConvAE)(meta["k"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, meta
