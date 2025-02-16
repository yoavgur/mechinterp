from typing import List, Tuple
from transformer_lens import HookedTransformer
import torch
import plotly.graph_objects as go
import plotly.express as px

# TODO this module should be connected to Interpreter objects

def generate(
    model: HookedTransformer,
    message: str, 
    force_output_prefix: str = None,
    max_new_tokens: int = 256,
    add_template_if_possible: bool = True,
) -> Tuple[List[str], str, str]:
    """
    Generates a response for `message` with `model`; 
    Returns: a tuple with
        (i) a list of token ids that includes the given message and generated response, wrapped in the chat template if added;
        (ii) the corresponding full string;
        (iii) the string of the generated response alone.

    Notes:
    - `message` is added with special tokens (such as <bos>) if needed.
    - If `add_template_if_possible`, then `model.tokenizer.chat_template`(if not None) is added for 
    wrapping the message prior to generation, including the tokens before generation. 
    - Currently generation is deterministic (`do_sample=False`); can be generalized in the future [TODO].
    - If `force_output_prefix` is not None, it is used as the output prefix.
    """

   # 1. Tokenization:
    # 1.A. Default tokenization:
    input_ids = model.tokenizer.encode(message, return_tensors="pt", add_special_tokens=True)
    start_of_gen_idx = input_ids.shape[-1]

    # 1.B. Chat-template tokenization:
    if add_template_if_possible and model.tokenizer.chat_template is not None:
        wrapped_message = [
            {"role": "user", "content": message},
        ]
        wrapped_message = model.tokenizer.apply_chat_template(wrapped_message, tokenize=False)
        start_of_gen_idx = model.tokenizer.encode(wrapped_message, add_special_tokens=False, return_tensors="pt").shape[-1]

        if force_output_prefix is not None:
            wrapped_message = wrapped_message + force_output_prefix
        
        input_ids = model.tokenizer.encode(wrapped_message, add_special_tokens=False, return_tensors="pt")

        if model.tokenizer.bos_token_id not in input_ids:
            print("[WARN] BOS token not found after adding chat template. \n This is unexpected. Consider adding the chat template manually")
            
    elif not add_template_if_possible and model.tokenizer.chat_template is not None:
        print("[WARN] Chat template is available, but not used. Use `add_template_if_possible=True` to use it.")
    elif add_template_if_possible and model.tokenizer.chat_template is None:
        print("[WARN] Chat template is not available, thus not used.")
    
    # 2. Generate:
    full_chat_toks = model.generate(
        input_ids,
        return_type="tensor",
        do_sample=False,
        max_new_tokens=max_new_tokens,
        prepend_bos=False,
    )
    full_chat_toks = full_chat_toks[0]  # remove batch dim

    # 3. Decode response:
    full_chat_str = model.tokenizer.decode(full_chat_toks, skip_special_tokens=False)
    # 3'. Trim just the response:
    response_toks = full_chat_toks[start_of_gen_idx:]
    response_str = model.tokenizer.decode(response_toks, skip_special_tokens=True)

    return (
        full_chat_toks.tolist(),
        full_chat_str,
        response_str,
    )


def cosine_with_direction(
        model: HookedTransformer,
        toks: List[int],
        direction: torch.Tensor,  # (hidden_size,)
        on_hook: str = 'resid',
) -> Tuple[go.Figure, torch.Tensor]: 
    """
    Measures cosine similarity of `model` activations (`on_hook`) with the given `direction`, for the input tokens `toks`.
    
    Available `on_hook` options: ['resid', 'attn', 'mlp', 'decomp_resid']; defaults to the final residuals ('resid').
    
    Returns: a tuple of 
    (i) a heatmap (of `go.Figure`) showing the similarities;
    (ii) a tensor with these similarities (n_hooks, seq_len).
    """

    # 1. Cache the relevant hidden states:
    assert on_hook in ['resid', 'attn_out', 'mlp_out', 'decomp_resid'], f"Unsupported hook: {on_hook}"
    on_hook_tl = {  # translate to TransformerLens hooks:
        'resid': 'blocks.{layer}.hook_resid_post',
        'attn': 'blocks.{layer}.hook_attn_out',
        'mlp': 'blocks.{layer}.hook_mlp_out',
        'decomp_resid': 'decompose_resid',
    }[on_hook]

    _, cache = model.run_with_cache(toks)
    if on_hook == 'decompose_resid':
        hidden_states, labels = cache.decompose_resid(return_labels=True)
        hidden_states = hidden_states.squeeze(1) # layer, seq, hidden
        n_layers = hidden_states.shape[0]
    else:
        n_layers = model.cfg.n_layers
        hidden_states = torch.stack([
            cache[on_hook_tl.format(layer=layer)] for layer in range(n_layers)
            ], dim=1).squeeze(0)  # layer, seq, hidden
        labels = [f"{on_hook}-l{layer}" for layer in range(n_layers)]

    # 2. Compute cosine similarity with activation:
    measures = torch.nn.functional.cosine_similarity(
        hidden_states,
        direction.to(hidden_states.device),
        dim=-1
    )

    # 3. Plot in heatmap, then show the plot
    fig = px.imshow(measures.cpu().numpy())
    
    hover_data = []
    toks_str = model.to_str_tokens(torch.tensor(toks))
    for layer in range(n_layers):
        hover_data.append([f"Cos: {measures[layer, i] :.3f} <br>Layer {layer} <br>Token: {tok}" for i, tok in enumerate(toks_str)])
    fig.update_traces(hovertemplate='%{customdata}', customdata=hover_data)
    
    fig.update_xaxes(tickvals=list(range(len(toks_str))), ticktext=toks_str, tickangle=50)
    fig.update_yaxes(title_text="Layer", tickvals=list(range(n_layers)), ticktext=labels)

    fig.show()
    
    return fig, measures


def rank_token_through_layers():
    # [TODO] track rank in of a given token in logit lens, through the layers; 
    #        visualize the rank (x-axis) as function of the layer (y-axis)
    raise NotImplementedError()