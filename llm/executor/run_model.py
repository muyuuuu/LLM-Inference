import torch

from .load_model import Qwen3Loader


def generate_no_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    device: str = "cuda",
):

    seq_len = len(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    all_input_ids = input_ids.clone()

    for step in range(max_new_tokens):
        current_len = all_input_ids.shape[1]
        offset = torch.arange(current_len, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(inputs=all_input_ids, offset=offset, is_causal=True)

        # get last token predict
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        if next_token.item() == tokenizer.eos_token_id:
            print("EOS token. Stopping.")
            break
        all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)

    decoded = tokenizer.decode(all_input_ids[0], skip_special_tokens=False)
    print(">>> In:", decoded[:seq_len])
    print(">>> Out:", decoded[seq_len:])


if __name__ == "__main__":
    loader = Qwen3Loader()
    qwen3_model, _, tokenizer = loader.convert_official_model()
    generate_no_cache(qwen3_model, tokenizer=tokenizer, prompt="hello, qwen3")
