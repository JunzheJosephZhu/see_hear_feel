from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from torch import nn
# train_nodes, eval_nodes = get_graph_node_names(resnet50(pretrained=True))
# print(train_nodes)
image_encoder = create_feature_extractor(resnet50(pretrained=True), ["layer4.2.add"])
test_image = torch.zeros((1, 3, 300, 300))
print(image_encoder(test_image))




def make_audio_extractor():
    audio_extractor = resnet50()
    audio_extractor.conv1 = nn.Conv2d(2, audio_extractor.inplanes, kernel_size=7, stride=2, padding=3, bias=False)