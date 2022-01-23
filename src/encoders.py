from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn

class Encoder(torch.nn.Module):
    def __init__(self, feature_extractor, out_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projection = nn.Linear(512, out_dim)

    def forward(self, x):
        feats = self.feature_extractor(x)["avgpool"]
        feats = feats.squeeze(3).squeeze(2)
        return self.projection(feats)

def make_vision_encoder(out_dim):
    vision_extractor = resnet18(pretrained=True)
    vision_extractor = create_feature_extractor(vision_extractor, ["avgpool"])
    return Encoder(vision_extractor, out_dim)


def make_tactile_encoder(out_dim):
    tactile_extractor = resnet18()
    tactile_extractor = create_feature_extractor(tactile_extractor, ["avgpool"])
    return Encoder(tactile_extractor, out_dim)


def make_audio_encoder(out_dim):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    audio_extractor = create_feature_extractor(audio_extractor, ["avgpool"])
    return Encoder(audio_extractor, out_dim)

if __name__ == "__main__":
    vision_encoder = make_vision_encoder(128)
    empty_input = torch.zeros((1, 3, 64, 101))
    print(vision_encoder(empty_input).shape)
