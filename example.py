import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

from torch.profiler import (
    profile,
    ProfilerActivity,
)


def main():
    path = os.path.expanduser("/root/autodl-tmp/models/Qwen3-0.6B/")

    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=16,
    )

    warmup_prompts = [
        "Explain what a compiler does.",
        "List several common sorting algorithms.",
    ]

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    warmup_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in warmup_prompts
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    llm.generate(warmup_prompts, sampling_params)

    # ============================
    # Start PyTorch Profiler
    # ============================
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        outputs = llm.generate(
            prompts,
            sampling_params,
        )

    # ============================
    # Export Chrome / Perfetto trace
    # ============================
    prof.export_chrome_trace("nano_vllm_trace.json")

    # ============================
    # Print operator statistics
    # ============================
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        )
    )

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
