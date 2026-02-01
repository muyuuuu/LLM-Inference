import torch

from .load_model import Qwen3Loader


def generate_no_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    device: str = "cuda",
):

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    all_input_ids = input_ids.clone()
    seq_len = all_input_ids.shape[1]

    for _ in range(max_new_tokens):
        current_len = all_input_ids.shape[1]
        offset = torch.arange(current_len, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            # logits = model(input_ids=all_input_ids, position_ids=offset).logits
            logits = model(inputs=all_input_ids, offset=offset)

        # get last token predict
        next_token_logits = logits[:, -1, :] / temperature
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            print("EOS token. Stopping.")
            break
        all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)

    input_tokens = all_input_ids[0, :seq_len]
    output_tokens = all_input_ids[0, seq_len:]

    print(">>> In:", tokenizer.decode(input_tokens, skip_special_tokens=True))
    print(">>> Out:", tokenizer.decode(output_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    loader = Qwen3Loader()
    qwen3_model, _, tokenizer = loader.convert_official_model()
    generate_no_cache(
        qwen3_model,
        tokenizer=tokenizer,
        prompt="what's deepseek, answer shortly.",
    )
