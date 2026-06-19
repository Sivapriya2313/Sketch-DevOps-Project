import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineResNet50(nn.Module):
    """
    Baseline SBIR model:
    - Uses ResNet50 backbone
    - Produces 512-d L2 normalized embeddings
    - No attention, no noise, no partial strokes
    """
    def __init__(self, emb_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Replace the classifier
        in_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.fc = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        if x.shape[1] == 1:  # grayscale sketches
            x = x.repeat(1, 3, 1, 1)

        feat = self.backbone(x)
        emb = self.fc(feat)
        emb = F.normalize(emb, dim=1)
        return emb