import torchvision.models as models
from torch import nn
import torch



class BasicResnet(nn.Module):
    def __init__(self, base_model, out_dim, pretrained):
        super(BasicResnet, self).__init__()
        if base_model == 'resnet18':
            self.base = models.resnet18(weights=None, num_classes=out_dim)
        elif base_model == 'resnet50':
            self.base = models.resnet50(weights=None, num_classes=out_dim)
        else:
            raise ValueError(base_model)
        self.base.fc = nn.Sequential()
    
    def forward(self, x):
        return self.base(x)
    
class LinearClassifier(nn.Module):
    def __init__(self, base_model, pretrained, num_class):
        super(LinearClassifier, self).__init__()
        if base_model == 'resnet18':
            base = models.resnet18(weights=None, num_classes=num_class)
        elif base_model == 'resnet50':
            base = models.resnet50(weights=None, num_classes=num_class)
        else:
            raise ValueError(base_model)
        dim_mlp = base.fc.in_features
        self.fc = nn.Linear(dim_mlp, num_class)
    
    def forward(self, features):
        return self.fc(features)
    
class ProjectionHead(nn.Module):
    def __init__(self, base_model, pretrained, out_dim):
        super(ProjectionHead,self).__init__()

        if base_model == 'resnet18':
            base = models.resnet18(weights=None, num_classes=out_dim)
        elif base_model == 'resnet50':
            base = models.resnet50(weights=None, num_classes=out_dim)
        else:
            raise ValueError(base_model)
        dim_mlp = base.fc.in_features

        # add mlp projection head
        self.pro_head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), base.fc)

    def forward(self, features):
        return self.pro_head(features)

class Student(nn.Module):
    def __init__(self, base_model, pro_dim, num_class, pretrained=False) -> None:
        super(Student, self).__init__()
        self.base_model = BasicResnet(base_model=base_model, out_dim=128, pretrained=pretrained)
        self.classifier = LinearClassifier(base_model=base_model, pretrained=pretrained, num_class=num_class)
        self.projection_head = ProjectionHead(base_model=base_model, pretrained=pretrained, out_dim=pro_dim)

    def forward(self, x):
        feature = self.base_model(x)
        class_logit = self.classifier(feature)
        sim_feature = self.projection_head(feature)
        return class_logit, sim_feature