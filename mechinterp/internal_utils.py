import string
from transformer_lens import HookedTransformer
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

def transpose_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(*reversed(range(tensor.ndim))).contiguous()

def align_tensor(tensor: InterpTensorType, d_model: int) -> Float[torch.Tensor, "... d_model"]:
    if tensor.shape[-1] == d_model:
        return tensor
    elif tensor.shape[0] == d_model:
        return transpose_tensor(tensor)
    else:
        raise ValueError(f"Tensor shape {tensor.shape} does not match d_model {d_model}")

def count_placeholders(s: str) -> int:
    formatter = string.Formatter()
    return sum(1 for _, field_name, _, _ in formatter.parse(s) if field_name is not None)

def split_by_placeholders(s: str) -> list[str]:
    formatter = string.Formatter()

    output = []
    output_str = ""

    for prefix, field_name, _, _ in formatter.parse(s):
        output_str += prefix

        if field_name is None:
            continue

        output.append(output_str)
        output_str = ""

    output.append(output_str)

    return output

def join_list(tok: torch.Tensor, lst: list[str], model: HookedTransformer, prepend_bos: bool = True) -> tuple[list[int], list[int]]:
    output = []
    indices = []

    if not lst:
        return output, indices
    
    bos = model.to_tokens("", prepend_bos=True)[0,0].item()

    if prepend_bos:
        output.append(bos)

    output.extend(model.to_tokens(lst[0], prepend_bos=False).tolist()[0])

    for item in lst[1:]:
        indices.append(len(output))
        output.append(tok)

        output.extend(model.to_tokens(item, prepend_bos=False).tolist()[0])

    return output, indices

def format_toks(model: HookedTransformer, prompt, placeholder_tok="X", prepend_bos: bool = True) -> tuple[torch.Tensor, list[int]]:
    tok = model.to_single_token(placeholder_tok)
    splits = split_by_placeholders(prompt)
    output, indices = join_list(tok, splits, model, prepend_bos=prepend_bos)
    return torch.tensor(output), indices

def get_model_identifier(model_name):
    model_name = model_name.lower()
    if model_name == "stable-vicuna":
        return "CarperAI/stable-vicuna-13b"
    if model_name.startswith("pythia"):
        return f"EleutherAI/{model_name}"
    elif model_name.startswith("gpt-neox"):
        return f"EleutherAI/{model_name}"
    elif model_name.startswith("llama-2"):
        return f"meta-llama/{model_name}"
    elif model_name.startswith("meta-llama"):
        return f"meta-llama/{model_name}"
    elif model_name.startswith("llama"):
        return f"facebook/{model_name}"
    elif model_name.startswith("opt"):
        return f"facebook/{model_name}"
    elif model_name.startswith("gpt2"):
        return model_name
    elif model_name.startswith("vicuna"):
        return f"lmsys/{model_name}"
    else:
        return model_name
