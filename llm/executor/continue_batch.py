import torch
from dataclasses import dataclass
from llm.executor.kv_cache import BatchKVCache, EasyKVCache
from llm.executor.load_model import Qwen3Loader


def topk_func(x, top_k=100):
    """
    filter the logits by top-k
    Args:
        x: (batch_size, vocab_size)
        top_k: int
    Returns:
        x: (batch_size, vocab_size)
    """
    top_k_values, _ = torch.topk(x, top_k, dim=-1)
    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)  # shape: (batch_size, 1)
    x[x < min_top_k_value] = -1e5
    return x


@dataclass
class RequestStatus:
    is_prompt = 0
    is_process = 1


class ModelRunner:
    def __init__(
        self,
        model,
        tokenizer,
        max_request_size: int = 1,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        filter=lambda x: x,
    ) -> None:
        self._max_request_size = max_request_size
        self._max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.filter = filter

    def run(
        self,
        input_ids: torch.Tensor,
        offset: torch.Tensor,
        cache,
        is_causal=True,
        mask=None,
    ):
        logits = self.model(
            input_ids, offset=offset, is_causal=is_causal, mask=mask, cache=cache
        )
        next_token_logits = logits[:, -1, :] / self.temperature
        next_token_logits = self.filter(next_token_logits)
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        return next_token


class SingleRequest:
    def __init__(
        self,
        prompt: str,
        chunk_size: int = 32,
        model_executor: ModelRunner = None,
    ) -> None:
        self._prompt = prompt
        self._chunk_size = chunk_size
        self._prefill_done = False
        self._decode_done = False
        self._model_executor = model_executor
        self._input_ids = model_executor.tokenizer.apply_chat_template(
            [{"role": "user", "content": self._prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).cuda()
        self._total_tokens_size = self._input_ids.shape[1]
        self._kv_cache = [
            EasyKVCache() for _ in range(model_executor.model.config.num_layers)
        ]
        self._next_token = None
        self._offset = 0
        self._infos = self._input_ids.tolist()
        self._infos = [x for row in self._infos for x in row]

    @property
    def next_token(self):
        return self._next_token

    @property
    def offset(self):
        return self._offset

    @property
    def kv_cache(self):
        return self._kv_cache

    @property
    def decode_done(self):
        return self._decode_done

    @property
    def prefill_done(self):
        return self._prefill_done

    def prefill(self):
        if self._prefill_done:
            return
        prefill_length = min(self._chunk_size, self._total_tokens_size - self._offset)
        self._next_token = self._model_executor.run(
            input_ids=self._input_ids[:, self._offset : self._offset + prefill_length],
            offset=torch.arange(
                self._offset,
                self._offset + prefill_length,
                dtype=torch.long,
                device=self._input_ids.device,
            ).unsqueeze(0),
            cache=self._kv_cache,
            is_causal=True,
            mask=None,
        )
        self._offset += prefill_length
        if self._offset == self._total_tokens_size:
            self._prefill_done = True
            self._infos.append(self._next_token.item())

    def decode(self, next_token):
        if not self._prefill_done:
            raise RuntimeError("Prefill not done")

        if self._decode_done:
            return

        self._next_token = next_token
        self._offset += 1
        if self._next_token.item() == self._model_executor.tokenizer.eos_token_id:
            self._decode_done = True
            print(
                self._model_executor.tokenizer.decode(
                    self._infos, skip_special_tokens=True
                )
            )

        self._infos.append(self._next_token.item())


class Executor:
    def __init__(self):
        self._promptes = [
            "黄金还会涨吗？简短回答",
            "千问是什么？简短回答",
            "hoka 运动鞋为什么不耐磨？简短回答",
            "python 的列表推导式怎么写？简短回答",
            "cursor 是什么？简短回答",
            "麦理浩径徒步路线？简短回答",
        ]

        # batch size is 3
        self.max_request_size = 3
        self.max_seq_len = 512
        filter = topk_func

        loader = Qwen3Loader()
        model, _, tokenizer = loader.convert_official_model(use_flash_attention=True)

        self.model_runner = ModelRunner(
            model=model,
            tokenizer=tokenizer,
            max_request_size=self.max_request_size,
            max_seq_len=self.max_seq_len,
            temperature=1.0,
            filter=filter,
        )

        self._batch_kv_cache = [
            BatchKVCache(
                max_activate_requests=self.max_request_size,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(model.config.num_layers)
        ]

    def run(self):
        all_slots_done = [RequestStatus.is_prompt] * self.max_request_size
        batches = [None] * self.max_request_size
        pending_prefill_request = None

        while 1:
            # end condition
            if len(self._promptes) == 0 and all(
                s == RequestStatus.is_prompt for s in all_slots_done
            ):
                break

            # get new prompt
            if len(self._promptes) > 0 and pending_prefill_request is None:
                prompt = self._promptes.pop(0)
                pending_prefill_request = SingleRequest(
                    prompt, model_executor=self.model_runner
                )

            # find processable request
            if pending_prefill_request is not None:
                if not pending_prefill_request.prefill_done:
                    pending_prefill_request.prefill()
                if pending_prefill_request.prefill_done:
                    find_next_request = False
                    for i in range(self.max_request_size):
                        if all_slots_done[i] == RequestStatus.is_prompt:
                            all_slots_done[i] = RequestStatus.is_process
                            for prefill, batch_cache in zip(
                                pending_prefill_request.kv_cache,
                                self._batch_kv_cache,
                            ):
                                batch_cache.add_request(prefill, request_idx=i)
                            find_next_request = True
                            batches[i] = pending_prefill_request
                            break
                    if find_next_request:
                        pending_prefill_request = None

            # decode
            if any(s == RequestStatus.is_process for s in all_slots_done):
                next_tokens = []
                offsets = []
                for batch in batches:
                    if batch is None:
                        next_tokens.append(0)
                        offsets.append(0)
                    else:
                        next_tokens.append(batch.next_token.item())
                        offsets.append(batch.offset)

                next_tokens = torch.tensor(next_tokens).cuda().view(-1, 1)
                offsets = torch.tensor(offsets).cuda().view(-1, 1)

                next_tokens = self.model_runner.run(
                    next_tokens,
                    offsets,
                    self._batch_kv_cache,
                    is_causal=False,
                    mask=None,
                )

                for i in range(self.max_request_size):
                    if all_slots_done[i] == RequestStatus.is_process:
                        batches[i].decode(next_tokens[i])
                        remove_reason = None
                        if batches[i].decode_done:
                            remove_reason = "decode done"
                        elif batches[i].offset >= self.max_seq_len:
                            remove_reason = "offset out of range"

                        if remove_reason is not None:
                            for cache in self._batch_kv_cache:
                                cache.remove_request(i)
                            all_slots_done[i] = RequestStatus.is_prompt
                            batches[i] = None
                            print(f"request {i} is done: {remove_reason}")


if __name__ == "__main__":
    executor = Executor()
    executor.run()
