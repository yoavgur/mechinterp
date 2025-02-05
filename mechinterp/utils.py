from typing import Union
from jaxtyping import Float
import torch

InterpTensorType = Union[Float[torch.Tensor, "... d_model"], Float[torch.Tensor, "d_model ..."]]

def reshape_list(lst, shape):
    total = 1
    for s in shape:
        total *= s
    if total != len(lst):
        raise ValueError("List length does not match shape")

    def helper(sub_lst, dims):
        if len(dims) == 1:
            return sub_lst
        sub_size = 1
        for d in dims[1:]:
            sub_size *= d
        return [helper(sub_lst[i * sub_size:(i + 1) * sub_size], dims[1:]) for i in range(dims[0])]

    return helper(lst, shape)

def recursive_flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(recursive_flatten(item))
        else:
            flattened.append(item)
    return flattened

def align_tensor(tensor: InterpTensorType, d_model: int) -> Float[torch.Tensor, "... d_model"]:
    if tensor.shape[-1] == d_model:
        return tensor
    elif tensor.shape[0] == d_model:
        return tensor.T
    else:
        raise ValueError(f"Tensor shape {tensor.shape} does not match d_model {d_model}")
