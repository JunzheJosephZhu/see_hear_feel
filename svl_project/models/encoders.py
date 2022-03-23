from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""
    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result





class Encoder(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Vision_Encoder(Encoder):
    def __init__(self, feature_extractor, out_dim):
        super().__init__(feature_extractor)
        self.ln = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.ln(x)
        return x

class Audio_Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.ln = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)["avgpool"]
        x = x.squeeze(3).squeeze(2)
        x = self.ln(x)
        return x

class Tactile_Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.ln = nn.Linear(512, out_dim)

    def forward(self, x):
        x = x[:, 2:-1, :, :] - x[:, 0:2, :, :]
        x = self.feature_extractor(x)["avgpool"]
        x = x.squeeze(3).squeeze(2)
        x = self.ln(x)
        return x

def make_vision_encoder(out_dim):
    vision_extractor = resnet18(pretrained=True)
    vision_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    return Vision_Encoder(vision_extractor, out_dim)

def make_vision_encoder_downsampled(conv_bottleneck, out_dim):
    vision_extractor = resnet18(pretrained=False)
    vision_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    return Encoder(vision_extractor, conv_bottleneck, out_dim, out_shape=(4, 5))

def make_tactile_encoder():
    tactile_extractor = resnet18(pretrained=True)
    tactile_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    tactile_extractor = create_feature_extractor(tactile_extractor, ["layer4.1.relu_1"])
    return Encoder(tactile_extractor)

def make_tactile_flow_encoder(out_dim):
    tactile_extractor = resnet18(pretrained=True)
    tactile_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    tactile_extractor = create_feature_extractor(tactile_extractor, ["avgpool"])
    return Tactile_Encoder(tactile_extractor, out_dim)

# @DeprecationWarning
def make_audio_encoder(conv_bottleneck, out_dim):
    audio_extractor = resnet18()
    audio_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
        ## vae_vision model is trained with 4 in channels
        # 4, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    audio_extractor = create_feature_extractor(audio_extractor, ["avgpool"])
    return Audio_Encoder(audio_extractor, out_dim)
    ## vae_vision model is trained with Encoder
    # return Encoder(audio_extractor)

if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_vision_encoder(64, 1280)
    print(encoder(inp).shape)