# 大模型推理

由于好奇大模型推理的实现细节，在 [tiny-llm](https://skyzh.github.io/tiny-llm/week1-07-sampling-prepare.html) 的基础上做了些改动，遂有了本仓库。实现主要技术，精力有限，所以不会做极致的性能优化。

## pre-requirements

- 需要 Nvidia 显卡，安装 cuda 环境，以及 pytorch
- 为测试代码准性，需要安装 torchao 和 torchtune
- 为了加载 huggingface 的 Qwen3 model，需要安装 transformers。如果网络失败，需要 `export HF_ENDPOINT=https://hf-mirror.com`

## QWen3 实现与回答生成

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
| ✅ Task 8: 实现 `Qwen3 TransformerBlock`                  | 暂时没想到测试方法                                                          |
| ✅ Task 9: 加载 Qwen3 模型，简单推理                      | `python -m llm.executor.run_model` 执行推理                                 |

```
Qwen3 Config Load Success
Qwen3 Tokenizer Load Success
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 46.81it/s]
Qwen3 Model Load Success
Convert QWen3 Model Success
>>> In: what's deepseek, answer shortly.
>>> Out:  DeepSeek is a large language model developed by DeepSeek, a company established in 2023. The model is trained on a large amount
```

## 工程优化

- sampling
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
- [vllm 中 Qwen3 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)