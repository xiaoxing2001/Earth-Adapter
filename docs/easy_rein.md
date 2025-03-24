# How to Integrate Rein into Your Existing Backbone?

This document demonstrates a straightforward method to integrate Rein into a new backbone. All code provided here relies solely on the PyTorch library.

## 1. Define Rein and LoRARein

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from torch import Tensor

class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(torch.empty([self.num_layers, self.token_length, self.embed_dims]))
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(6.0/ 3.0 * self.patch_size*self.patch_size + self.embed_dims)
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:            
            return self.learnable_tokens # return all
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(feats,tokens,layer)
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        attn = attn * (self.embed_dims**-0.5)
        attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

```

## 2. Use Rein in the Backbone

1. **Modify the backbone's initialization function to build Rein:**
```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Assume num_layers=12, embed_dims=768, patch_size=16
    self.reins = LoRAReins(12, 768, 16)
```

2. **Modify the backbone's forward function to use Rein:**
```python
def forward(self, x):
    # Code from the original forward function
    for idx, layer in enumerate(self.layers):
        x = layer(x)
        # Set batch_first to True if the shape of x is (B, N, C), else False
        # Set has_cls_token to True if the cls token is in x, else False
        x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
    # Assume the final features needed is 'outs'
    return self.reins.return_auto(outs)
```

3. **Modify the train function to make Rein the only trainable parameter:**
```python
import torch.nn as nn
from typing import List

first_set_requires_grad = True
first_set_train = True

def set_requires_grad(model: nn.Module, keywords: List[str]):
    """
    Set parameters to require gradients based on keyword inclusion in their names.
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
        else:
            param.requires_grad = False
    global first_set_requires_grad
    if first_set_requires_grad:
        first_set_requires_grad = False

def _set_train(model: nn.Module, keywords: List[str], prefix: str = ""):
    train_names = []
    for name, child in model.named_children():
        fullname = ".".join([prefix, name])
        if any(name.startswith(key) for key in keywords):
            train_names.append(fullname)
            child.train()
        else:
            train_names += _set_train(child, keywords, prefix=fullname)
    return train_names

def set_train(model: nn.Module, keywords: List[str]):
    """
    Set submodules to training mode based on keyword startswith condition.
    """
    model.train(False)
    train_names = _set_train(model, keywords)
    global first_set_train
    if first_set_train:
        first_set_train = False

def train(self, mode: bool = True):
    if not mode:
        return super().train(mode)
    set_requires_grad(self, ["reins"])
    set_train(self, ["reins"])
```

4. **Ensure that your optimizer only adds parameters that need training.**