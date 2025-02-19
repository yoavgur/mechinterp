import os
import sys
from contextlib import nullcontext
from typing import Tuple
import torch
import pytest
from transformer_lens import HookedTransformer
from jaxtyping import Float
# So that we import the local version of mechinterp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mechinterp import Interpreter, InterpTensorType

D_MODEL = 1024 # gpt2-medium
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

@pytest.fixture(scope="session")
def model_name() -> str:
    return "gpt2-medium"

@pytest.fixture(scope="session")
def model(model_name: str):
    return HookedTransformer.from_pretrained(model_name, device=device)

def get_v(shape: tuple[int, ...]) -> InterpTensorType:
    return torch.randn(shape, device=device)

@pytest.mark.parametrize("input_token, top_tokens, bottom_tokens", [
    ("hello", ['hello', 'Hello', ' hello', ' neighb'], [' Force', ',', ' further', ' rig']),
    (" abstract", [' abstract', ' abstraction', ' Abstract', ' mathemat'], [' North', ' Mark', ' Bet', ' Jimmy']),
])
def test_logit_lens_fuck(model: HookedTransformer, input_token: str, top_tokens: list[str], bottom_tokens: list[str]):
    interp = Interpreter(model)

    v = model.embed(model.to_single_token(input_token))
    ll = interp.logit_lens(v, k=len(top_tokens))
    print(ll)

    assert ll.top == top_tokens
    assert ll.bottom == bottom_tokens

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

@pytest.fixture(scope="session")
def pt_vector(model: HookedTransformer) -> Float[torch.Tensor, "d_model"]:
    prompt = "Apple's former CEO attended the Oscars"
    return model.run_with_cache(prompt)[1]["blocks.18.hook_resid_pre"][0, model.get_token_position(" CEO", prompt)]

@pytest.mark.parametrize("shape", [(1,), (1, 2, 3, 4), (2, D_MODEL, 3)])
def test_logit_lens_bad_shape(model: HookedTransformer, shape: Tuple[int, ...]):
    interp = Interpreter(model)
    v = get_v(shape)
    with pytest.raises(ValueError):
        interp.logit_lens(v, k=10)

def test_patchscopes(model: HookedTransformer, pt_vector: Float[torch.Tensor, "d_model"]):
    interp = Interpreter(model)
    assert interp.patchscopes(pt_vector, "The birth name of {} is '", temperature=0, target_layer=3, n=3, prepend_bos=False).explanation == "The birth name of X is 'Steve Jobs'"

@pytest.mark.parametrize("prompt, shape, exception", [
    ("Bla {} bla", (D_MODEL,), nullcontext()),
    ("Bla {} bla", (1, D_MODEL,), nullcontext()),
    ("Bla {} bla", (1, 1, D_MODEL,), nullcontext()),
    ("Bla {} bla", (1, 1, 1, 1, 1, D_MODEL,), nullcontext()),
    ("Bla {} bla", (1, 2, 3, 4), pytest.raises(ValueError)),
    ("Bla {} bla", (1, 2, 3, 4, D_MODEL), pytest.raises(AssertionError)),

    ("Bla bla", (D_MODEL,), pytest.raises(AssertionError)),
    ("Bla {} {} bla", (D_MODEL,), pytest.raises(AssertionError)),
    ("{}", (D_MODEL,), nullcontext()),
    ("{} aaa", (D_MODEL,), nullcontext()),
    ("aaa {}", (D_MODEL,), nullcontext()),

    ("Bla {} bla {}", (2,D_MODEL), nullcontext()),
    ("Bla {} bla {} aaa", (2,D_MODEL), nullcontext()),
    ("{} Bla bla {}", (2,D_MODEL), nullcontext()),
    ("{}{}", (2,D_MODEL), nullcontext()),

    ("Bla {} {} {} b la", (1,D_MODEL), pytest.raises(AssertionError)),
    ("Bla {} {} {} b la", (2,D_MODEL), pytest.raises(AssertionError)),
    ("Bla {} {} {} b la", (3,D_MODEL), nullcontext()),
    ("Bla {} {} {} b la", (4, 3,D_MODEL), pytest.raises(AssertionError)),
])
def test_patchscopes_shapes_and_prompts(model: HookedTransformer, prompt: str, shape: Tuple[int, ...], exception):
    interp = Interpreter(model)
    v = get_v(shape)

    with exception:
        interp.patchscopes(v, prompt, n=2)