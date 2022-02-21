from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(BasicBlock, self).__init__()
        if deconv:
            self.conv1 = deconv(in_planes, planes, kernel_size=3, stride=stride, padding=1, output_padding=0 if stride==1 else 1)
            self.conv2 = deconv(planes, planes, kernel_size=3, stride=1, padding=1)
            self.deconv = True
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.deconv = False


        self.shortcut = nn.Sequential()

        if not deconv:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            #self.bn1 = nn.GroupNorm(planes//16,planes)
            #self.bn2 = nn.GroupNorm(planes//16,planes)

            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    #nn.GroupNorm(self.expansion * planes//16,self.expansion * planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = deconv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, output_padding=1)

    def forward(self, x):

        if self.deconv:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out

        else: #self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

class ResNet_Decoder(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], deconv=nn.ConvTranspose2d, out_channels=None, in_shape=(10, 8)):
        super(ResNet_Decoder, self).__init__()
        self.in_shape = in_shape
        self.in_linear = nn.Linear(1280, 64 * in_shape[0]* in_shape[1])
        self.in_planes = 512

        self.deconv = True
        self.conv1 = nn.ConvTranspose2d(64, 512, kernel_size=3, stride=2, padding=1, output_padding=(1, 0))


        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=2, deconv=deconv)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2, deconv=deconv)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, deconv=deconv)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2, deconv=deconv)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deconv))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.in_linear(x)
        out = out.reshape(x.size(0), 64, self.in_shape[0], self.in_shape[1])
        out = F.relu(self.conv1(out))
        out = F.upsample_bilinear(out, scale_factor=(2, 2))
        print(out.shape)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out)
        print(out.shape)
        out = self.out_conv(out)
        print(out.shape)
        return out

def make_vision_decoder():
    return ResNet_Decoder(out_channels=3)

def make_audio_decoder():
    return ResNet_Decoder(out_channels=2)

def make_tactile_decoder():
    return ResNet_Decoder(out_channels=3)

