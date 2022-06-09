from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, feature_extractor, conv_bottleneck, out_dim, out_shape):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.conv1x1 = nn.Conv2d(512, conv_bottleneck, 1)
        self.projection = nn.Linear(conv_bottleneck * out_shape[0] * out_shape[1], out_dim)

    def forward(self, x):
        feats = self.feature_extractor(x)
        assert len(feats.values()) == 1
        feats = list(feats.values())[0]
        feats = self.downsample(feats)
        feats = self.conv1x1(feats)
        feats = feats.flatten(-3, -1)
        feats = self.projection(feats)
        return feats

def make_vision_encoder(conv_bottleneck, out_dim):
    vision_extractor = resnet18(pretrained=True)
    # change the first conv layer to fit 30 channels
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    return Encoder(vision_extractor, conv_bottleneck, out_dim, out_shape=(7, 10))

@DeprecationWarning
def make_tactile_encoder(out_dim):
    tactile_extractor = resnet18()
    tactile_extractor = create_feature_extractor(tactile_extractor, ["avgpool"])
    return Encoder(tactile_extractor, out_dim)

@DeprecationWarning
def make_audio_encoder(out_dim):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    audio_extractor = create_feature_extractor(audio_extractor, ["avgpool"])
    return Encoder(audio_extractor, out_dim)

if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_vision_encoder(64, 1280)
    print(encoder(inp).shape)