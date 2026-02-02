import torch


def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    batch_size = logits.shape[0]
    for i in range(batch_size):
        for token_id in set(input_ids[i].tolist()):
            if logits[i, token_id] > 0:
                logits[i, token_id] /= penalty
            else:
                logits[i, token_id] *= penalty
    return logits


def generate_no_cache_base(
    model,
    tokenizer,
    input_ids,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    device: str = "cuda",
    filter=lambda x: x,
):
    all_input_ids = input_ids.clone()
    seq_len = all_input_ids.shape[1]

    for _ in range(max_new_tokens):
        current_len = all_input_ids.shape[1]
        offset = torch.arange(current_len, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(inputs=all_input_ids, offset=offset)

        # get last token predict
        next_token_logits = logits[:, -1, :] / temperature
        next_token_logits = filter(next_token_logits)

        apply_repetition_penalty(next_token_logits, all_input_ids)

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


def generate_topk_nocache(
    model,
    tokenizer,
    input_ids,
    top_k=100,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    device: str = "cuda",
):

    def topk_func(x):
        top_k_values, _ = torch.topk(x, top_k, dim=-1)
        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)  # shape: (batch_size, 1)
        x[x < min_top_k_value] = -float("inf")
        return x

    generate_no_cache_base(
        model, tokenizer, input_ids, max_new_tokens, temperature, device, topk_func
    )


def generate_topn_nocache(
    model,
    tokenizer,
    input_ids,
    top_p=0.8,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    device: str = "cuda",
):

    def topn_func(x):
        sorted_logits, sorted_indices = torch.sort(x, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        x[indices_to_remove] = -float("inf")
        return x

    generate_no_cache_base(
        model, tokenizer, input_ids, max_new_tokens, temperature, device, topn_func
    )
