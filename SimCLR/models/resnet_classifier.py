import torch.nn as nn
import torchvision.models as models
from exceptions.exceptions import InvalidBackboneError

class ResNetClassifier(nn.Module):
    def __init__(self, base_model, num_class):
        super(ResNetClassifier, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=num_class),
                            "resnet50": models.resnet50(weights=None, num_classes=num_class)}
        self.base_model = self._get_basemodel(base_model)
    
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model
        
    def forward(self, x):
        return self.base_model(x)

if __name__ == '__main__':
    print(1)