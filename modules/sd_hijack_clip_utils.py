"""Compatibility helpers for transformers 4.x vs 5.x CLIPTextModel layout."""

SD1_CLIP_WEIGHT_KEYS = (
    'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight',
    'cond_stage_model.transformer.embeddings.token_embedding.weight',
)


def is_flat_clip_transformer(transformer):
    return hasattr(transformer, 'embeddings') and not hasattr(transformer, 'text_model')


def clip_text_embeddings(transformer):
    text_model = getattr(transformer, 'text_model', None)
    if text_model is not None:
        return text_model.embeddings
    return transformer.embeddings


def clip_text_final_layer_norm(transformer):
    text_model = getattr(transformer, 'text_model', None)
    if text_model is not None:
        return text_model.final_layer_norm
    return transformer.final_layer_norm


def state_dict_includes_sd1_clip(state_dict):
    return any(key in state_dict for key in SD1_CLIP_WEIGHT_KEYS)


def remap_sd1_clip_state_dict_for_flat_clip_text_model(state_dict):
    """Map checkpoint keys transformer.text_model.* -> transformer.* for flattened CLIPTextModel."""
    prefix = 'cond_stage_model.transformer.text_model.'
    flat_prefix = 'cond_stage_model.transformer.'

    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            remapped[flat_prefix + key[len(prefix):]] = value
        else:
            remapped[key] = value

    return remapped
