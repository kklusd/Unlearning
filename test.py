import torch
from torchvision import transforms, datasets
from SimCLR.models.resnet_classifier import ResNetClassifier
from mu.mu_models import ProjectionHead, BasicResnet, Student
from SimCLR.models.resnet_simclr import ResNetSimCLR
import numpy as np
import copy
base_model = 'resnet18'
unlearn_teacher = ResNetClassifier(base_model, num_class=10)
compete_teacher = ResNetClassifier(base_model, num_class=10)
student = Student(base_model, pretrained=False, num_class=10, pro_dim=128)
teacher_path = './SimCLR/runs/original_model/checkpoint_0200.pth.tar'
simCLR_path = './SimCLR/runs/sim_model/checkpoint_0020.pth.tar'
checkpoint = torch.load(teacher_path, map_location=torch.device('cpu'))
checkpoint_sim = torch.load(simCLR_path, map_location=torch.device('cpu'))
student_state = copy.deepcopy(student.state_dict())
compete_teacher.load_state_dict(checkpoint['state_dict'])
simCLR = ResNetSimCLR(base_model, out_dim=128)
simCLR.load_state_dict(checkpoint_sim['state_dict'])
base_resnet = BasicResnet(base_model, out_dim=10, pretrained=False)
print(len(student.state_dict()), len(compete_teacher.state_dict()))
for param_tensor in student.state_dict():
    print(param_tensor, "\t", student.state_dict()[param_tensor].size())
for i in range(len(compete_teacher.state_dict().keys())):
    key_1 = list(compete_teacher.state_dict().keys())[i]
    key_2 = list(student.state_dict().keys())[i]
    student_state[key_2] = copy.deepcopy(compete_teacher.state_dict()[key_1])
# # student.load_state_dict(student_state)
for i in range(1, 5):
    key_1 = list(simCLR.state_dict().keys())[-i]
    key_2 = list(student.state_dict().keys())[-i]
    student_state[key_2] = copy.deepcopy(simCLR.state_dict()[key_1])
student.load_state_dict(student_state)
# print(key_2, "\t", student.state_dict()[key_2])
# print(key_1, "\t", simCLR.state_dict()[key_1])
for param_tensor in student.state_dict():
    print(param_tensor, "\t", student.state_dict()[param_tensor].size())
# for param_tensor in compete_teacher.state_dict():
#     print(param_tensor, "\t", compete_teacher.state_dict()[param_tensor].size())
# print('===========')
# print(compete_teacher.state_dict())
# for param_tensor in base_resnet.state_dict():
#     print(param_tensor, "\t", base_resnet.state_dict()[param_tensor].size())


# train_ds = datasets.CIFAR10(root='SimCLR/datasets', train=True, download=True)
# test_ds = datasets.CIFAR10(root='SimCLR/datasets', train=False)
# for i in range(len(train_ds)):
#     print(train_ds[i])
# print(np.arange(0, 10))
# for k, v in student.named_parameters():
#     if 'projection_head' in k.split('.'):
#         print('unfreezing %s' % k)
#         v.requires_grad_(False)
#     print(k, v)
a = (1, 2)

