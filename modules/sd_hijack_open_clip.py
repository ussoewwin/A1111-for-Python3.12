import open_clip.tokenizer
import torch

from modules import sd_hijack_clip, devices
from modules.shared import opts

tokenizer = open_clip.tokenizer._tokenizer


class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder["<start_of_text>"]
        self.id_end = tokenizer.encoder["<end_of_text>"]
        self.id_pad = 0

    def tokenize(self, texts):
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'

        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized

    def encode_with_transformers(self, tokens):
        # set self.wrapped.layer_idx here according to opts.CLIP_stop_at_last_layers
        z = self.wrapped.encode_with_transformer(tokens)

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)

        return embedded


class FrozenOpenCLIPEmbedder2WithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder["<start_of_text>"]
        self.id_end = tokenizer.encoder["<end_of_text>"]
        self.id_pad = 0

    def tokenize(self, texts):
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'

        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized

    def encode_with_transformers(self, tokens):
        # Fix for PyTorch 2.8+ attn_mask shape issue
        original_attn_mask = None
        if hasattr(self.wrapped.model, 'attn_mask') and self.wrapped.model.attn_mask is not None:
            # Store original mask
            original_attn_mask = self.wrapped.model.attn_mask
            # Fix shape if it's [77, 77] - convert to broadcastable shape
            if original_attn_mask.shape == (77, 77):
                self.wrapped.model.attn_mask = None  # Temporarily disable mask
        
        try:
            d = self.wrapped.encode_with_transformer(tokens)
            z = d[self.wrapped.layer]

            pooled = d.get("pooled")
            if pooled is not None:
                z.pooled = pooled

            return z
        finally:
            # Restore original mask
            if original_attn_mask is not None:
                self.wrapped.model.attn_mask = original_attn_mask

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids.to(self.wrapped.model.token_embedding.wrapped.weight.device)).squeeze(0)

        return embedded
