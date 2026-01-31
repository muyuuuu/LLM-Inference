import torch.nn.functional as F


from llm.ops import (
    _MLP,
    _RMSNorm,
    _Rope,
    _scaled_dot_product_attention,
)


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        *,
        hidden_size,
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
        max_seq_len=32768,
        theta=1e6,
        rms_norm_eps=1e-5,
    ):
        assert hidden_size % q_num_head == 0, "hidden_size % q_num_head != 0"
        assert q_num_head % kv_num_head == 0, "q_num_head % kv_num_head != 0"

        self.hidden_size = hidden_size
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

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

        self.rope_layer = _Rope(
            self.head_dim, self.max_seq_len, self.theta, traditional=False
        )

    def __call__(self, x, *, mask=None):
        batch_size, seq_len, _ = x.size()
        _q = F.Linear(x, self.q_weight, self.q_bias).reshape(
            batch_size, seq_len, self.q_num_head, self.head_dim
        )
        _k = F.Linear(x, self.k_weight, self.k_bias).reshape(
            batch_size, seq_len, self.q_num_head, self.head_dim
        )
        _v = F.Linear(x, self.v_weight, self.v_bias).reshape(
            batch_size, seq_len, self.q_num_head, self.head_dim
        )

        _q = self.rope_layer(_q)
        _k = self.rope_layer(_k)

        _q = _q.transpose(0, 2, 1, 3)
        _k = _k.transpose(0, 2, 1, 3)
        _v = _v.transpose(0, 2, 1, 3)

        y = (
            _scaled_dot_product_attention(_q, _k, _v, mask=mask)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.hidden_size)
        )

        return F.linear(y, self.o_weight)


class Qwen2TransformBlock:
    def __init__(
        self,
        *,
        hidden_size,  # args for multiherad attention
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
        max_seq_len=32768,  # args for rope
        theta=1e6,
        w_gate,  # args for mlp
        w_up,
        w_down,
        w_rms_norm1,  # args for rms_norm
        w_rms_norm2,
        rms_norm_eps=1e-5,
    ):
        self.attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
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
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
        )

        self.mlp = _MLP(hidden_size, w_gate, w_up, w_down)
        self.rms_norm1 = _RMSNorm(hidden_size, w_rms_norm1, eps=rms_norm_eps)
        self.rms_norm2 = _RMSNorm(hidden_size, w_rms_norm2, eps=rms_norm_eps)

    def __call__(self, x, *, mask=None):
        y = self.attn(self.rms_norm1(x), mask)
        r1 = x + y
        r2 = self.mlp(self.rms_norm2(r1))
        return r1 + r2
