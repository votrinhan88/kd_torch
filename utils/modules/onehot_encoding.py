import torch

class OneHotEncoding(torch.nn.Module):
    def __init__(self, num_classes:int, dtype:torch.dtype=torch.float):
        super().__init__()
        self.num_classes = num_classes
        self.dtype = dtype
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes)
        x = x.squeeze(dim=-2)
        x = x.to(dtype=self.dtype)
        return x
