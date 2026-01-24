# 大模型推理

## pre-requires

1. cuda，pytorch

## 算子实现

- [x] Task 1: 实现 scaled_dot_product_attention。 `python -m unittest llm.test.attention_test.TestScaleDotAttention`
- [x] Task 2: 实现 MultiHeadAttention。`python -m unittest llm.test.attention_test.TestMultiHeadAttention`
- [x] Task 3: 实现 RoPE 旋转位置编码。`python -m unittest llm/test/rope_test.py`

# 参考

- [tiny-llm](https://github.com/skyzh/tiny-llm)
- [torch MultiHeadAttention 转置](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5945)
- [torch 实现 RoPE](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py)