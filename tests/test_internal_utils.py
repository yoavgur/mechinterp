import os
import sys
import pytest
import torch
from contextlib import nullcontext
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mechinterp.internal_utils import transpose_tensor, recursive_flatten, align_tensor, count_placeholders, split_by_placeholders, join_list, format_toks

D_MODEL = 1024 # gpt2-medium
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

@pytest.fixture(scope="session")
def model_name() -> str:
    return "gpt2-medium"

@pytest.fixture(scope="session")
def model(model_name: str):
    return HookedTransformer.from_pretrained(model_name, device=device)

@pytest.mark.parametrize(
    "input_tensor, expected_tensor",
    [
        # Test case 1: 2D tensor
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 3], [2, 4]])),
        # Test case 2: 3D tensor
        (torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), torch.tensor([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])),
        # Test case 3: 1D tensor (should be no change)
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])),
        # Test case 4: Empty tensor
        (torch.tensor([]), torch.tensor([])),
        # Test case 5: Tensor with different data type (e.g., float)
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 3.0], [2.0, 4.0]])),
    ],
)
def test_transpose_tensor(input_tensor, expected_tensor):
    transposed_tensor = transpose_tensor(input_tensor)
    assert torch.equal(transposed_tensor, expected_tensor)

@pytest.mark.parametrize(
    "input_list, expected_flattened_list",
    [
        ([1, [2, 3], 4, [5, [6, 7]]], [1, 2, 3, 4, 5, 6, 7]), # Simple nested list
        ([1, "a", [2, "b"], 3.0, [["c"]]], [1, "a", 2, "b", 3.0, "c"]), # List with mixed types
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), # Already flat list
        ([], []), # Empty list
        ([1, [2, [3, [4, [5]]]]], [1, 2, 3, 4, 5]), # Deeply nested list
    ],
)
def test_recursive_flatten(input_list, expected_flattened_list):
    assert recursive_flatten(input_list) == expected_flattened_list

@pytest.mark.parametrize(
    "input_shape, d_model, expected_output, exception",
    [
        ((5, 10), 10, (5, 10), nullcontext()),
        ((10, 5), 10, (5, 10), nullcontext()),
        ((2, 10, 30, 40, 20), 2, (20, 40, 30, 10, 2), nullcontext()),
        ((5, 20), 10, None, pytest.raises(ValueError)),
        ((20, 5), 10, None, pytest.raises(ValueError)),
        ((7, 10, 8), 10, None, pytest.raises(ValueError)),
    ],
)
def test_align_tensor(input_shape, d_model, expected_output, exception):
    input_tensor = torch.randn(input_shape)
    with exception:
        aligned_tensor = align_tensor(input_tensor, d_model)
        assert aligned_tensor.shape == expected_output

@pytest.mark.parametrize(
    "input_string, expected_count",
    [
        ("No placeholders", 0),
        ("One placeholder: {}", 1),
        ("Two placeholders: {} and {}", 2),
        ("Nested placeholders: {{}} and {{{}}}", 1),
        ("{}{}{}", 3),
        ("", 0),
    ],
)
def test_count_placeholders(input_string, expected_count):
    assert count_placeholders(input_string) == expected_count

@pytest.mark.parametrize(
    "input_string, expected_splits",
    [
        ("No placeholders", ["No placeholders"]),
        ("One placeholder: {}", ["One placeholder: ", ""]),
        ("Two placeholders: {} and {}", ["Two placeholders: ", " and ", ""]),
        ("Placeholders at start {} and end {}", ["Placeholders at start ", " and end ", ""]),
        ("Nested placeholders: {{}} and {{{}}} suffix", ["Nested placeholders: {} and {", "} suffix"]),
        ("Mixed text and placeholders: Start {} middle {} end", ["Mixed text and placeholders: Start ", " middle ", " end"]),
        ("", [""]),
        ("}}", ["}"]),
        ("{{}}", ["{}"]),
    ],
)
def test_split_by_placeholders(input_string, expected_splits):
    assert split_by_placeholders(input_string) == expected_splits

@pytest.mark.parametrize(
    "input_list, prepend_bos, expected_output",
    [
        ([], True, ([], [])),
        (["a b c"], True, ([50256, 64, 275, 269], [])),
        (["a b c", "d e f"], True, ([50256, 64, 275, 269, 55, 67, 304, 277], [4])),
        (["a b c", "d e f"], False, ([64, 275, 269, 55, 67, 304, 277], [3])),
        (["a b c", "d e f", "d e f"], True, ([50256, 64, 275, 269, 55, 67, 304, 277, 55, 67, 304, 277], [4, 8])),
        (["a b c", "d e f", "d e f"], False, ([64, 275, 269, 55, 67, 304, 277, 55, 67, 304, 277], [3, 7])),
    ],
)
def test_join_list(model, input_list, prepend_bos, expected_output):
    tok = model.to_single_token("X")
    result_output, result_indices = join_list(tok, input_list, model, prepend_bos=prepend_bos)
    assert (result_output, result_indices) == expected_output

@pytest.mark.parametrize(
    "prompt, placeholder_tok, prepend_bos, expected_output",
    [
        ("a b c", "X", True, ([50256, 64, 275, 269], [])), # tokens for "a b c"
        ("a b c{}", "X", True, ([50256, 64, 275, 269, 55], [4])), # tokens for "a b cX"
        ("a b c{}d e f", "X", False, ([64, 275, 269, 55, 67, 304, 277], [3])), # tokens for "a b cXd e f"
        ("{}a b c{}d e f", "X", False, ([55, 64, 275, 269, 55, 67, 304, 277], [0,4])), # tokens for "a b cXd e f"
        ("z{}a b c{}d e f", "X", False, ([89, 55, 64, 275, 269, 55, 67, 304, 277], [1,5])), # tokens for "a b cXd e f"
        ("{}{}{}{}", "X", False, ([55,55,55,55], [0,1,2,3])), # tokens for "a b cXd e f"
        ("{}{}{}{}", "X", True, ([50256, 55, 55, 55, 55], [1,2,3,4])), # tokens for "a b cXd e f"
    ],
)
def test_format_toks(model, prompt, placeholder_tok, prepend_bos, expected_output):
    result_output, result_indices = format_toks(model, prompt, placeholder_tok=placeholder_tok, prepend_bos=prepend_bos)
    assert (result_output.tolist(), result_indices) == expected_output
