from ..mu_models import BasicResnet
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import tqdm
import os

class OwnDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index][0]
        return x

def feature_model_loader(base_model, out_dim, state_dict_path, device):
    new_resnet = BasicResnet(base_model=base_model, out_dim=out_dim)
    check_point = torch.load(state_dict_path, map_location=device)
    check_point_state = check_point['state_dict']
    base_model_state = new_resnet.state_dict()
    for i in range(len(base_model_state.keys())):
        key_1 = list(check_point_state.keys())[i]
        key_2 = list(base_model_state.keys())[i]
        base_model_state[key_2] = copy.deepcopy(check_point_state[key_1])
    new_resnet.load_state_dict(base_model_state)
    new_resnet.to(device)
    return new_resnet


class features_generator:
    def __init__(self, base_model, out_dim, state_dict_path, device):
        model = feature_model_loader(base_model, out_dim, state_dict_path, device)
        self.model = model.eval()

    def generate(self, forget_data, retain_data, batch_size, out_path):
        forget_set = OwnDataset(forget_data)
        retain_set = OwnDataset(retain_data)
        forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=False, 
                                   num_workers=2, pin_memory=True)
        retain_loader = DataLoader(retain_set, batch_size=batch_size, shuffle=False, 
                                  num_works=2, pin_memory=True)
        forget_features = []
        retain_features = []
        for batch in tqdm(forget_loader, desc='forget features generate',leave=False):
            x = batch
            features = self.model(x)
            forget_features.append(features)
        for batch in tqdm(retain_loader, desc='retain features generate', leave=False):
            x = batch
            features = self.model(x)
            retain_features.append(features)
        forget_features = torch.cat(forget_features, dim=0).cpu()
        torch.save(forget_features, os.path.join(out_path, 'forget_features.pt'))
        retain_features = torch.cat(retain_features, dim=0).cpu()
        torch.save(retain_features, os.path.join(out_path, 'retain_features.pt'))
        return forget_features, retain_features