import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
        self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, out_size),
                            )
    def forward(self, x):
        out = self.model(x)
        if self.norm_reduce:
            out = torch.norm(out)

        return out

class ContextNetEx(nn.Module):

    def __init__(self, input_shape, out_channels, hidden_dim, kernel_size, norm_type='batch'):
        super(ContextNetEx, self).__init__()

        # Keep same dimensions
        in_channels, h, w = input_shape
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding)

        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm([hidden_dim, h, w])
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm2d(hidden_dim)
        else:
            self.norm1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)

        if norm_type == 'layer':
            self.norm2 = nn.LayerNorm([hidden_dim, h, w])
        elif norm_type == 'instance':
            self.norm2 = nn.InstanceNorm2d(hidden_dim)
        else:
            self.norm2 = nn.BatchNorm2d(hidden_dim)

        self.final = nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=padding)

        if norm_type == 'layer':
            self.norm3 = nn.LayerNorm([hidden_dim, h, w])
        elif norm_type == 'instance':
            self.norm3 = nn.InstanceNorm2d(hidden_dim)
        else:
            self.norm3 = nn.BatchNorm2d(hidden_dim)



    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        out = self.final(x)
        out = self.norm3(out)
        out = F.relu(x)
        return out

class ContextNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (kernel_size - 1) // 2

        self.context_net = nn.Sequential(
                                nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=padding),
                                # nn.BatchNorm2d(out_channels), # added
                                # nn.ReLU(), # added
                            )


    def forward(self, x):
        out = self.context_net(x)
        return out


class ConvNetUNC(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, smaller_model=True, hidden_dim=128, return_features=False, dropout_rate=0.3, **kwargs):
        super(ConvNetUNC, self).__init__()

        kernel_size = 5
        self.smaller_model = smaller_model
        self.dropout_rate = dropout_rate
        print("dropout_rate is", self.dropout_rate)
        padding = (kernel_size - 1) // 2
    

        if smaller_model:
            print("using smaller model")

            self.conv1 = nn.Sequential(
                            nn.Conv2d(num_channels, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(p=self.dropout_rate),
                            nn.MaxPool2d(2)
                        )
        else:
            print("using larger model")
            self.conv0 = nn.Sequential(
                        nn.Conv2d(num_channels, hidden_dim, kernel_size, padding=padding),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout_rate),                        
                    )
            
            self.conv1 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(p=self.dropout_rate),
                            nn.MaxPool2d(2)
                        )            
            
        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout_rate),
                        nn.MaxPool2d(2)
                        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.final = nn.Sequential(
                    nn.Linear(hidden_dim, 200),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_rate),
                    Identity() if return_features else nn.Linear(200, num_classes)
                  )
        self.num_features = 200


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""

        # x shape: batch_size, num_channels, w, h
        if self.smaller_model:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)

        out = self.conv2(out)
        out = self.adaptive_pool(out) # shape: batch_size, hidden_dim, 1, 1
        out = out.squeeze(dim=-1).squeeze(dim=-1) # make sure not to squeeze the first dimension when batch size is 0.
        out = self.final(out)

        return out


class ResNetContext(nn.Module):

    def __init__(self, input_shape, in_channels, out_channels, model_name, pretrained=None,
                 avgpool=False):
        super(ResNetContext, self).__init__()

        if pretrained:
            weights = 'ResNet50_Weights.DEFAULT'

        self.input_shape = input_shape
        self.model = torchvision.models.__dict__[model_name](weights=weights)
        self.num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(self.num_features, out_channels)

        # Change number of input channels from 3 to whatever is needed
        # to take in the context also.
        model_inplanes = 64
        old_weights = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(in_channels, model_inplanes,
                            kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            for i in range(in_channels):
                self.model.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]

        if avgpool:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.model(x)
        c, h, w = self.input_shape
        out = out.view(-1, c, h, w)
        return out






class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, smaller_model=True, hidden_dim=128, return_features=False, **kwargs):
        super(ConvNet, self).__init__()

        kernel_size = 5

        self.smaller_model = smaller_model
        padding = (kernel_size - 1) // 2
        if smaller_model:
            print("using smaller model")
            self.conv1 = nn.Sequential(
                            nn.Conv2d(num_channels, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
        else:
            print("using larger model")
            self.conv0 = nn.Sequential(
                        nn.Conv2d(num_channels, hidden_dim, kernel_size, padding=padding),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                    )

            self.conv1 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )


        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.final = nn.Sequential(
                    nn.Linear(hidden_dim, 200),
                    nn.ReLU(),
                    Identity() if return_features else nn.Linear(200, num_classes)
                  )
        self.num_features = 200


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""

        # x shape: batch_size, num_channels, w, h

        if self.smaller_model:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)
        out = self.conv2(out)
        out = self.adaptive_pool(out) # shape: batch_size, hidden_dim, 1, 1
        out = out.squeeze(dim=-1).squeeze(dim=-1) # make sure not to squeeze the first dimension when batch size is 0.
        out = self.final(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_channels, num_classes, model_name, pretrained=None,
                 avgpool=False, return_features=False):
        super(ResNet, self).__init__()

        if pretrained:
            weights = 'ResNet50_Weights.DEFAULT'
            # weights = 'ResNet50_Weights.IMAGENET1K_V1'

        self.model = torchvision.models.__dict__[model_name](weights=weights)
        self.num_features = self.model.fc.in_features
        if return_features:
            self.model.fc = Identity()
        else:
            self.model.fc = nn.Linear(self.num_features, num_classes)

        # Change number of input channels from 3 to whatever is needed
        # to take in the context also.
        if num_channels != 3:
            model_inplanes = 64
            old_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(num_channels, model_inplanes,
                             kernel_size=7, stride=2, padding=3, bias=False)

            if pretrained:
                for i in range(num_channels):
                    self.model.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]

        if avgpool:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out = self.model(x)
        return out



class ResNet_UNC(nn.Module):
    def __init__(self, num_channels, num_classes, model_name, pretrained=None,
                 avgpool=False, return_features=False, dropout_rate=0.2):
        super(ResNet_UNC, self).__init__()

        if pretrained:
            weights = 'ResNet50_Weights.DEFAULT'
            # weights = 'ResNet50_Weights.IMAGENET1K_V1'
            

        model_name = model_name.split('_')[0]
        self.model = torchvision.models.__dict__[model_name](weights=weights)
        self.num_features = self.model.fc.in_features
        if return_features:
            self.model.fc = Identity()
        else:
            self.model.fc = nn.Linear(self.num_features, num_classes)

        # Change number of input channels from 3 to whatever is needed
        # to take in the context also.
        if num_channels != 3:
            model_inplanes = 64
            old_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(num_channels, model_inplanes,
                             kernel_size=7, stride=2, padding=3, bias=False)

            if pretrained:
                for i in range(num_channels):
                    self.model.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]

        if avgpool:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.model.layer1.dropout = nn.Dropout(p=dropout_rate)
        self.model.layer2.dropout = nn.Dropout(p=dropout_rate)
        self.model.layer3.dropout = nn.Dropout(p=dropout_rate)
        self.model.layer4.dropout = nn.Dropout(p=dropout_rate)

        for name, module in self.model.named_modules():
            print(name)

    def forward(self, x):
        out = self.model(x)
        return out
