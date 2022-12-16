from collections import OrderedDict
from typing import Tuple, Union

import torch

class PlaceholderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_batches:int=120,
        batch_size:int=512,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        return (torch.zeros([self.batch_size, 1]), torch.zeros([self.batch_size, 1]))

class IntermediateFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model:torch.nn.Module,
        out_layers:dict[str, torch.nn.Module],
        with_output=True,
    ):  
        super().__init__()
        self.model = model
        self.out_layers = out_layers
        self.with_output = with_output
        
        self.features = OrderedDict({k: None for k in self.out_layers.keys()})
        self.handles = OrderedDict({k: None for k in self.out_layers.keys()})

        for key, layer in self.out_layers.items():
            def hook(module, input, output, key=key):
                if self.features[key] is None:
                    self.features[key] = output
                else:
                    if isinstance(self.features[key], list):
                        self.features[key].append(output)
                    else:
                        self.features[key] = [self.features[key], output]
            h = layer.register_forward_hook(hook)
            self.handles[key] = h

    def __call__(self, *args, **kwargs) -> Union[Tuple[dict, torch.Tensor], torch.Tensor]:
        self.features = OrderedDict({k: None for k in self.out_layers.keys()})
        output = self.model(*args, **kwargs)
        
        if self.with_output:
            return self.features, output
        else:
            return self.features
    
    def remove_hooks(self):
        for h in self.handles.values():
            h.remove()
        self.handles.clear()

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from models.classifiers import LeNet5

    def test_feature_extractor():
        print(' Test IntermediateFeatureExtractor '.center(80,'#'))
        IMAGE_DIM = [1, 32, 32]
        BATCH_SIZE = 128
        NUM_CLASSES = 10
        
        net = LeNet5(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        out_layers = OrderedDict({
            'flatten': net.flatten,
            'F6': net.F6,
            'logits': net.logits,
        })
        net = IntermediateFeatureExtractor(net, out_layers)

        x = torch.normal(mean=0, std=1, size=(BATCH_SIZE, *IMAGE_DIM))
        features, output = net(x)

        if (  (len(features)             == len(out_layers))
            & (features['flatten'].shape == torch.Size([BATCH_SIZE, 120]))
            & (features['F6'].shape      == torch.Size([BATCH_SIZE, 84]))
            & (features['logits'].shape  == torch.Size([BATCH_SIZE, NUM_CLASSES]))
            & (output.shape              == torch.Size([BATCH_SIZE, NUM_CLASSES]))
        ):
            net.remove_hooks()
            print('PASSED')
    
    test_feature_extractor()