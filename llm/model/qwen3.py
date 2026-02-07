import torch.nn.functional as F

from dataclasses import dataclass

from llm.layer import (
    _MLP,
    _RMSNorm,
    _Rope,
    _scaled_dot_product_attention,
    _TiedEmbedding,
)


@dataclass
class Qwen3Config:
    vocab_size: int = -1
    emb_out_hidden_size: int = -1
    q_num_head: int = -1
    kv_num_head: int = -1
    head_dim: int = -1
    num_layers: int = -1
    max_seq_len: int = -1
    theta: int = 1e6
    rms_eps: float = 1e-5
    use_tie_embedding: bool = False
    is_causal: bool = True


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        *,
        q_num_head,
        kv_num_head,
        head_dim,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        q_norm,
        k_norm,
        max_seq_len=32768,
        theta=1e6,
        rms_norm_eps=1e-5,
    ):
        assert q_num_head % kv_num_head == 0, "q_num_head % kv_num_head != 0"

        self.hidden_size = q_num_head * head_dim
        self.q_num_head = q_num_head
        self.kv_num_head = kv_num_head
        self.head_dim = head_dim

        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight

        self.q_bias = q_bias
        self.k_bias = k_bias
        self.v_bias = v_bias

        self.q_norm = _RMSNorm(q_norm, eps=rms_norm_eps)
        self.k_norm = _RMSNorm(k_norm, eps=rms_norm_eps)

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

        self.rope_layer = _Rope(
            self.head_dim,
            self.max_seq_len,
            self.theta,
            traditional=False,
            device="cuda",
        )

    def __call__(self, x, offset=None, is_causal=True, mask=None, cache=None):
        batch_size, seq_len, _ = x.size()
        _q = F.linear(x, self.q_weight, self.q_bias).reshape(
            batch_size, seq_len, self.q_num_head, self.head_dim
        )
        _k = F.linear(x, self.k_weight, self.k_bias).reshape(
            batch_size, seq_len, self.kv_num_head, self.head_dim
        )
        _v = F.linear(x, self.v_weight, self.v_bias).reshape(
            batch_size, seq_len, self.kv_num_head, self.head_dim
        )

        _q = self.q_norm(_q)
        _k = self.k_norm(_k)

        _q = self.rope_layer(_q, offset=offset)
        _k = self.rope_layer(_k, offset=offset)

        _q = _q.transpose(1, 2)
        _k = _k.transpose(1, 2)
        _v = _v.transpose(1, 2)

        if cache is not None:
            cache.save_kv_to_cache(_k, _v)
            _k, _v = cache.get_cached_kv()

        y = (
            _scaled_dot_product_attention(_q, _k, _v, is_causal=is_causal, mask=mask)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return F.linear(y, self.o_weight)


class Qwen2TransformBlock:
    def __init__(
        self,
        *,
        q_num_head,  # args for multiherad attention
        kv_num_head,
        head_dim,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        q_norm,
        k_norm,
        max_seq_len=32768,  # args for rope
        theta=1e6,
        w_gate,  # args for mlp
        w_up,
        w_down,
        w_rms_norm1,  # args for rms_norm
        w_rms_norm2,
        rms_norm_eps=1e-5,
    ):
        self.attn = Qwen3MultiHeadAttention(
            q_num_head=q_num_head,
            kv_num_head=kv_num_head,
            head_dim=head_dim,
            q_weight=q_weight,
            q_bias=q_bias,
            k_weight=k_weight,
            k_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            o_weight=o_weight,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = _MLP(w_gate, w_up, w_down)
        self.rms_norm1 = _RMSNorm(w_rms_norm1, eps=rms_norm_eps, device="cuda")
        self.rms_norm2 = _RMSNorm(w_rms_norm2, eps=rms_norm_eps, device="cuda")

    def __call__(self, x, offset=None, is_causal=True, mask=None, cache=None):
        y = self.attn(
            self.rms_norm1(x),
            offset=offset,
            is_causal=is_causal,
            mask=mask,
            cache=cache,
        )
        r1 = x + y
        r2 = self.mlp(self.rms_norm2(r1))
        return r1 + r2


class Qwen3Model:
    def __init__(self, config: Qwen3Config):
        self.config = config
        self.load_pretrained = False

    def from_pretrained(self, model):
        self.load_pretrained = True

        state = model.state_dict()

        self.emb_tokens = _TiedEmbedding(
            self.config.vocab_size,
            self.config.emb_out_hidden_size,
            state["model.embed_tokens.weight"],
        )

        self.layers = []

        for i in range(self.config.num_layers):
            q_proj_str = f"model.layers.{str(i)}.self_attn.q_proj.weight"
            k_proj_str = f"model.layers.{str(i)}.self_attn.k_proj.weight"
            v_proj_str = f"model.layers.{str(i)}.self_attn.v_proj.weight"
            o_proj_str = f"model.layers.{str(i)}.self_attn.o_proj.weight"
            q_norm_str = f"model.layers.{str(i)}.self_attn.q_norm.weight"
            k_norm_str = f"model.layers.{str(i)}.self_attn.k_norm.weight"
            gate_str = f"model.layers.{str(i)}.mlp.gate_proj.weight"
            up_str = f"model.layers.{str(i)}.mlp.up_proj.weight"
            down_str = f"model.layers.{str(i)}.mlp.down_proj.weight"
            norm_1_str = f"model.layers.{str(i)}.input_layernorm.weight"
            norm_2_str = f"model.layers.{str(i)}.post_attention_layernorm.weight"

            self.layers.append(
                Qwen2TransformBlock(
                    q_num_head=self.config.q_num_head,
                    kv_num_head=self.config.kv_num_head,
                    head_dim=self.config.head_dim,
                    q_weight=state[q_proj_str],
                    q_bias=None,
                    k_weight=state[k_proj_str],
                    k_bias=None,
                    v_weight=state[v_proj_str],
                    v_bias=None,
                    o_weight=state[o_proj_str],
                    q_norm=state[q_norm_str],
                    k_norm=state[k_norm_str],
                    w_gate=state[gate_str],
                    w_up=state[up_str],
                    w_down=state[down_str],
                    w_rms_norm1=state[norm_1_str],
                    w_rms_norm2=state[norm_2_str],
                    max_seq_len=self.config.max_seq_len,
                    theta=self.config.theta,
                    rms_norm_eps=self.config.rms_eps,
                )
            )

        self.last_norm = _RMSNorm(state["model.norm.weight"], eps=self.config.rms_eps)
        self.w_last_head = state["lm_head.weight"]

    def __call__(self, inputs, offset=None, is_causal=True, mask=None, cache=None):
        assert self.load_pretrained == True, "please load model first"

        y = self.emb_tokens(inputs)
        if cache is not None:
            for idx, layer in enumerate(self.layers):
                y = layer(
                    x=y, offset=offset, is_causal=is_causal, mask=mask, cache=cache[idx]
                )
        else:
            for layer in self.layers:
                y = layer(x=y, offset=offset, is_causal=is_causal, mask=mask)

        y = self.last_norm(y)
        return F.linear(y, self.w_last_head)
