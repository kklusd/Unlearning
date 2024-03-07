import torch.nn as nn
import torchvision.models as models
import torch
from ..exceptions.exceptions import InvalidBackboneError
from collections import OrderedDict


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


#for KD
class ResNetKD(nn.Module):

    def __init__(self):
        super(ResNetKD, self).__init__()
        self.classifier = nn.Sequential(OrderedDict([('fc2', nn.Linear(128, 10)),
                                                 ('output', nn.Softmax(dim=1))
                                                 ]))
        #
        checkpoint = torch.load('../networkparams/checkpoint_0020.pth.tar')  # 加载模型
        self.model = ResNetSimCLR('resnet18', 128)
        self.model.load_state_dict(checkpoint['state_dict'])


    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
