<div align="center">

# 🚀 LLM Inference

**从零实现大模型推理框架 — Qwen3 的本地部署与优化**

*侧重 GPU 优化技术，完成 Qwen3-4B-Instruct 的本地部署与推理加速*

</div>

---

基于对大模型推理原理的兴趣，本项目跟随 [tiny-llm](https://skyzh.github.io/tiny-llm/week1-07-sampling-prepare.html) 课程，将苹果芯片相关实现适配到 Nvidia GPU（RTX 4070），完成 **Qwen3-4B-Instruct-2507** 的本地部署与推理。显存不足时可选用 **Qwen3-0.5B**，重点在于学习与复现技术路线。

<div align="center">
  <img src="./assert/demo.png" width="640" alt="demo">
</div>

---

## 📋 Prerequisites

- **Nvidia GPU**：需安装 CUDA 及对应版本 PyTorch
- **验证与依赖**：`torchao`、`torchtune`（用于测试与对齐）
- **FlashAttention**：需安装 `triton`
- **模型加载**：`transformers`（国内可设 `export HF_ENDPOINT=https://hf-mirror.com`）

**推荐环境：**

| 依赖         | 版本        |
| ------------ | ----------- |
| torch        | 2.6.0+cu126 |
| torchao      | 0.15.0      |
| torchtune    | 0.6.1       |
| transformers | 4.57.6      |
| triton       | 3.2.0       |

---

## 📐 Qwen3 实现与本地部署

| #   | 任务                             | 测试命令                                                                   |
| --- | -------------------------------- | -------------------------------------------------------------------------- |
| 1   | ✅ `scaled_dot_product_attention` | `python -m unittest llm.test.attention_test.TestScaleDotAttention`         |
| 2   | ✅ `MultiHeadAttention`           | `python -m unittest llm.test.attention_test.TestMultiHeadAttention`        |
| 3   | ✅ RoPE 旋转位置编码              | `python -m unittest llm.test.rope_test`                                    |
| 4   | ✅ RMSNorm                        | `python -m unittest llm.test.norm_test`                                    |
| 5   | ✅ 千问 MLP                       | `python -m unittest llm.test.mlp_test`                                     |
| 6.1 | ✅ Attention 添加 GQA             | `python -m unittest llm.test.attention_test.TestScaleDotAttention`         |
| 6.2 | ✅ MultiHeadAttention GQA         | `python -m unittest llm.test.attention_test.TestGroupedMultiHeadAttention` |
| 7   | ✅ Tied Embedding                 | `python -m unittest llm.test.tie_embedding_test.TestTieEmbedding`          |
| 8   | ✅ Qwen3 TransformerBlock         | —                                                                          |
| 9   | ✅ 加载 Qwen3 并简单推理          | `python -m llm.executor.run_model`                                         |

---

## ⚡ 工程优化

| #   | 优化                             | 测试命令                                                                                         |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1.1 | ✅ TopK 采样                      | `python -m llm.executor.run_model --topk 100`                                                    |
| 1.2 | ✅ TopP 采样                      | `python -m llm.executor.run_model --topp 0.7`                                                    |
| 2   | ✅ Prefill-Decode 分离与 KV Cache | `python -m llm.executor.run_model --kv_cache 1`                                                  |
| 3   | ✅ FlashAttention V1 (Triton)     | `python -m unittest llm.test.flash_attn_test` · `python -m llm.executor.run_model --use_flash 1` |
| 4   | ✅ 连续批处理与 Chunk Prefill     | `python -m llm.executor.continue_batch`                                                          |
| 5   | ✅ PagedAttention                 | `python -m llm.executor.continue_batch --use_page 1`                                             |

---

## 📖 结语

大模型对话还会涉及许多 [优化与工程方向](https://www.bilibili.com/video/BV1Bm6bB5EJ3/?spm_id_from=333.337.search-card.all.click&vd_source=08fc039ce87a61f2dd6954658b5ae2b5)，更偏大模型服务与 Agent，例如：

- **Speculative Decoding**：小模型草稿 + 大模型并行验证
- **Memory**：长期 / 短期记忆管理
- **RAG**：检索增强生成
- **MCP (Model Context Protocol)**：与外部数据源、工具的协议
- **Skills**：多工具编排与 Agent 行为定义

RAG 涉及向量检索与数据库，Memory 涉及摘要与历史压缩，各家实现差异较大，与 GPU 推理优化关系相对独立。本仓库主要聚焦**单卡推理与 KV/Attention 优化**。后续回去学多卡相关的内容。

---

## 📚 参考

- [tiny-llm](https://github.com/skyzh/tiny-llm)
- [PyTorch MultiHeadAttention](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5945)
- [torchtune RoPE](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py)
- [RoPE 公式推导](https://spaces.ac.cn/archives/8265/comment-page-1)
- [PyTorch RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [Qwen2 MLP (transformers)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [GQA 简介](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-attention-and-grouped-query-attention/)
- [Tie Embedding](https://www.spaces.ac.cn/archives/9698)
- [vLLM Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [大模型推理：采样与惩罚](https://zhuanlan.zhihu.com/p/1981752176578667658)
- [图解 KV Cache](https://zhuanlan.zhihu.com/p/662498827)
- [Triton 入门](https://zhuanlan.zhihu.com/p/684473453)
- [FlashAttention 推导](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [Chunk Prefill](https://zhuanlan.zhihu.com/p/14689463165)
- [连续批处理](https://zhuanlan.zhihu.com/p/719610083)
- [PageAttention](https://zhuanlan.zhihu.com/p/691038809)
