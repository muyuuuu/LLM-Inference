# 大模型推理

## pre-requirements

- 需要 Nvidia 显卡，安装 cuda 环境，以及 pytorch。为测试代码准性，需要安装 torchao 和 torchtune

## QWen2 实现

| 任务                                                     | 测试命令                                                                    |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| ✅ Task 1: 实现 `scaled_dot_product_attention`            | `python -m unittest llm.test.attention_test.TestScaleDotAttention`          |
| ✅ Task 2: 实现 `MultiHeadAttention`                      | `python -m unittest llm.test.attention_test.TestMultiHeadAttention`         |
| ✅ Task 3: 实现 `RoPE` 旋转位置编码                       | `python -m unittest llm.test.rope_test`                                     |
| ✅ Task 4: 实现 `RMSNorm` 标准化                          | `python -m unittest llm.test.norm_test`                                     |
| ✅ Task 5: 实现千问的 `MLP`                               | `python -m unittest llm.test.mlp_test`                                      |
| ✅ Task 6.1: `scaled_dot_product_attention` 添加 GQA 支持 | `python -m unittest llm.test.attention_test.TestScaleDotAttention`          |
| ✅ Task 6.2: `MultiHeadAttention` 添加 GQA 支持           | `python -m unittest llm.test.attention_test.TestGroupedMultiHeadAttention ` |
| ✅ Task 7: 实现 `tied embedding`                          | `python -m unittest llm.test.tie_embedding_test.TestTieEmbedding`           |

## 工程优化

- key-value cache
- flash attention
- continuous batching

## 服务设计

......

# 参考

- [tiny-llm](https://github.com/skyzh/tiny-llm)
- [torch MultiHeadAttention 转置](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5945)
- [torch 实现 RoPE](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py)
- [RoPE 公式推导](https://spaces.ac.cn/archives/8265/comment-page-1)
- [torch 实现 RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [Qwen 实现 MLP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [GQA 概念](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-attention-and-grouped-query-attention/)
- [Tie Embedding](https://www.spaces.ac.cn/archives/9698)