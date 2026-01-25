# 大模型推理

## pre-requirements

1. cuda，pytorch

## 算子实现


| 任务                                          | 测试命令                                                            |
| --------------------------------------------- | ------------------------------------------------------------------- |
| ✅ Task 1: 实现 `scaled_dot_product_attention` | `python -m unittest llm.test.attention_test.TestScaleDotAttention`  |
| ✅ Task 2: 实现 `MultiHeadAttention`           | `python -m unittest llm.test.attention_test.TestMultiHeadAttention` |
| ✅ Task 3: 实现 `RoPE` 旋转位置编码            | `python -m unittest llm.test.rope_test`                             |
| ✅ Task 4: 实现 `RMSNorm` 标准化               | `python -m unittest llm.test.norm_test`                             |



# 参考

- [tiny-llm](https://github.com/skyzh/tiny-llm)
- [torch MultiHeadAttention 转置](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5945)
- [torch 实现 RoPE](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py)
- [RoPE 公式推导](https://www.bilibili.com/video/BV1CQoaY2EU2/?spm_id_from=333.337.search-card.all.click&vd_source=08fc039ce87a61f2dd6954658b5ae2b5)
- [torch 实现 RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)