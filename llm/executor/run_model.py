import argparse

from .load_model import Qwen3Loader
from .generate import (
    generate_no_cache_base,
    generate_topk_nocache,
    generate_topn_nocache,
    generate_kv_cache_base,
)


if __name__ == "__main__":
    loader = Qwen3Loader()
    qwen3_model, _, tokenizer = loader.convert_official_model()

    messages = [{"role": "system", "content": "什么是 qwen？不超过 20 字"}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).cuda()

    parser = argparse.ArgumentParser(
        description="Text generation with sampling strategies."
    )
    parser.add_argument(
        "--topk", type=int, default=100, help="Top-k sampling. If not set, disabled."
    )
    parser.add_argument(
        "--topp",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling. If not set, disabled.",
    )
    parser.add_argument(
        "--kv_cache",
        type=bool,
        default=False,
        help="use cache",
    )

    args = parser.parse_args()

    if args.kv_cache:
        generate_kv_cache_base(
            qwen3_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
        )
    else:
        if args.topk is None and args.topp is None:
            generate_no_cache_base(
                qwen3_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
            )
        else:
            if args.topk is not None:
                assert 100 <= args.topk <= 200
                generate_topk_nocache(
                    model=qwen3_model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    top_k=int(args.topk),
                )
            elif args.topp is not None:
                assert 0 < args.topp < 1.0
                generate_topn_nocache(
                    qwen3_model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    top_p=float(args.topp),
                )
