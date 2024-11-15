import torch
import math
from torch import nn
from transformers import CLIPTokenizer, T5TokenizerFast
from modules import sd_hijack


#################################################################################################
### Core/Utility
#################################################################################################

class AutocastLinear(nn.Linear):
    """Linear layer that casts its weights to the parameter type."""

    def forward(self, x):
        return torch.nn.functional.linear(
            x, 
            self.weight.to(x.dtype), 
            self.bias.to(x.dtype) if self.bias is not None else None
        )


def attention(q, k, v, heads, mask=None):
    """Wrapper for scaled dot-product attention."""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = [t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """Multi-layer perceptron (MLP) used in various architectures."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


#################################################################################################
### CLIP
#################################################################################################

class CLIPAttention(nn.Module):
    """Attention mechanism for CLIP model."""
    
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


class CLIPLayer(nn.Module):
    """Layer of the CLIP model consisting of attention and feedforward components."""
    
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = Mlp(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation], dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEncoder(nn.Module):
    """Encoder for the CLIP model, consisting of multiple layers."""
    
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, intermediate_output=None):
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(nn.Module):
    """Embeddings for the CLIP model, including token and position embeddings."""
    
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, textual_inversion_key="clip_l"):
        super().__init__()
        self.token_embedding = sd_hijack.TextualInversionEmbeddings(vocab_size, embed_dim, dtype=dtype, device=device, textual_inversion_key=textual_inversion_key)
        self.position_embedding = nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(nn.Module):
    """Text model for CLIP, including embeddings and encoder."""
    
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device, textual_inversion_key=config_dict.get('textual_inversion_key', 'clip_l'))
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[torch.arange(x.shape[0], device=x.device), input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1)]
        return x, i, pooled_output


class CLIPTextModel(nn.Module):
    """Main CLIP text model, wrapping the embedding and encoder."""
    
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class SDTokenizer:
    """Tokenizer for the SD model."""
    
    def __init__(self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True, min_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer('')["input_ids"]
        self.tokens_start = 1 if has_start_token else 0
        self.start_token = empty[0] if has_start_token else None
        self.end_token = empty[1]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str):
        """Tokenize the text with weight values."""
        pad_token = self.end_token if self.pad_with_end else 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x]
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):
    """Tokenizer for the SDXL Clip G model."""
    
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class SD3Tokenizer:
    """Tokenizer for the SD3 model, integrating multiple tokenizers."""
    
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text: str):
        """Tokenize text with weights for different models."""
        out = {
            "g": self.clip_g.tokenize_with_weights(text),
            "l": self.clip_l.tokenize_with_weights(text),
            "t5xxl": self.t5xxl.tokenize_with_weights(text)
        }
        return out


class ClipTokenWeightEncoder:
    """Encoder for token weights."""
    
    def encode_token_weights(self, token_weight_pairs):
        tokens = [a[0] for a in token_weight_pairs[0]]
        out, pooled = self([tokens])
        first_pooled = pooled[0:1].cpu() if pooled is not None else None
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(nn.Module, ClipTokenWeightEncoder):
    """CLIP transformer encoder for text."""
    
    LAYERS = ["last", "pooled", "hidden"]
    
    def __init__(self, device="cpu", max_length=77, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=CLIPTextModel, special_tokens=None, layer_norm_hidden_state=True, return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens or {"start": 49406, "end": 49407, "pad": 49407}
        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None and abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = torch.asarray(tokens, dtype=torch.int64, device=backup_embeds.weight.device)
        outputs = self.transformer(tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)
        z = outputs[0] if self.layer == "last" else outputs[1]
        pooled_output = outputs[2].float() if len(outputs) >= 3 and outputs[2] is not None else None
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface."""
    
    def __init__(self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience."""
    
    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5)