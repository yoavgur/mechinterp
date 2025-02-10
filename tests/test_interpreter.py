import os
import sys
import torch
import pytest
from transformer_lens import HookedTransformer

# So that we import the local version of mechinterp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mechinterp import Interpreter, InterpTensorType

D_MODEL = 768 # gpt2-small
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

@pytest.fixture(scope="session")
def model_name() -> str:
    return "gpt2-small"

@pytest.fixture(scope="session")
def model(model_name: str):
    return HookedTransformer.from_pretrained(model_name, device=device)

def get_v(shape: tuple[int, ...]) -> InterpTensorType:
    return torch.randn(shape, device=device)

@pytest.mark.parametrize("input_token, top_tokens, bottom_tokens", [
    ("hello", ['hey', 'hello', ' Mara', ' hello'], ['negie', ' advoc', 'earcher', 'ibli']),
    ("world", ['wide', 'ship', 'side', 's'], ['ournal', 'interstitial', 'emale', ' corrid']),
])
def test_logit_lens(model: HookedTransformer, input_token: str, top_tokens: list[str], bottom_tokens: list[str]):
    interp = Interpreter(model)

    v = model.embed(model.to_tokens(input_token, prepend_bos=False)[0,0])
    ll = interp.logit_lens(v, k=len(top_tokens), use_first_mlp=True)

    assert ll.topk == top_tokens
    assert ll.bottomk == bottom_tokens

@pytest.mark.parametrize("shape, k", [
    ((D_MODEL,), 10),
    ((1, D_MODEL), 10),
    ((1, 1, D_MODEL), 10),
    ((1, 1, 1, 1, 1, 1, 1, 1, 1, D_MODEL), 10),
    ((2, 1, 3, 1, 5, 1, D_MODEL), 10),
    ((3, D_MODEL), 5),
    ((5, 2, D_MODEL), 12),
    ((5, 2, 3, D_MODEL), 30),
    ((5, 20, 3, 4, D_MODEL), 1),
    ((5, 2, 3, 4, 5, D_MODEL), 7),
    ((D_MODEL, 1), 2),
    ((D_MODEL, 2, 1, 2, 1), 2),
    ((D_MODEL, 2, 3), 4),
    ((D_MODEL, 2, 3, 4, 5, 6), 2),
])
def test_logit_lens_shape(model: HookedTransformer, shape: tuple[int, ...], k: int):
    interp = Interpreter(model)

    v = get_v(shape)
    ll = interp.logit_lens(v, k=k)

    if shape[-1] == D_MODEL:
        assert ll.shape == (*[x for x in shape[:-1] if x != 1], k)
    else:
        assert ll.shape == (*reversed([x for x in shape[1:] if x != 1]), k)
