from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn
# from perceiver_pytorch import Perceiver
import torch.nn.functional as F
import torchaudio

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
    def __init__(self, feature_extractor, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))        
        if out_dim is not None:
            self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x

class Spec_Encoder(Encoder):
    def __init__(self, feature_extractor, out_dim=None, norm_audio=False):
        super().__init__(feature_extractor, out_dim)
        self.norm_audio = norm_audio
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64
        )

    def forward(self, waveform):
        EPS = 1e-8
        spec = self.mel(waveform.float())
        log_spec = torch.log(spec + EPS)
        assert log_spec.size(-2) == 64
        if self.norm_audio:
            log_spec /= log_spec.sum(dim=-2, keepdim=True) # [1, 64, 100]
        return super().forward(log_spec)

class Tactile_Flow_Encoder(nn.Module):
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

def make_vision_encoder(out_dim=None):
    vision_extractor = resnet18(pretrained=True)
    vision_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    vision_extractor = create_feature_extractor(vision_extractor, ["layer4.1.relu_1"])
    # return Vision_Encoder(vision_extractor, out_dim)
    return Encoder(vision_extractor, out_dim)

def make_tactile_encoder(out_dim):
    tactile_extractor = resnet18(pretrained=True)
    tactile_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    tactile_extractor = create_feature_extractor(tactile_extractor, ["layer4.1.relu_1"])
    return Encoder(tactile_extractor, out_dim)

def make_flow_encoder():
    input_dim = 2 * 10 * 14
    encoder = nn.Sequential(nn.Flatten(1), nn.Linear(input_dim, 2048),
                            nn.Linear(2048, 1024),
                            nn.Linear(1024, 512))
    return encoder

def make_tactile_flow_encoder(out_dim):
    tactile_extractor = resnet18(pretrained=False)
    tactile_extractor.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    tactile_extractor = create_feature_extractor(tactile_extractor, ["avgpool"])
    return Tactile_Flow_Encoder(tactile_extractor, out_dim)


def make_audio_encoder(out_dim=None, norm_audio=False):
    audio_extractor = resnet18(pretrained=True)
    audio_extractor.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    audio_extractor = create_feature_extractor(audio_extractor, ["layer4.1.relu_1"])
    return Spec_Encoder(audio_extractor, out_dim, norm_audio)

if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_vision_encoder(64, 1280)
    print(encoder(inp).shape)