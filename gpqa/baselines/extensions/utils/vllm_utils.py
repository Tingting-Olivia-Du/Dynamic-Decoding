"""vLLM 工具。vllm 为可选依赖，仅在调用 generate_vllm/build_llm/sample 时需要。"""

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False


def _require_vllm():
    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vllm is required for this function. Install with: pip install vllm"
        )


# Create an offline LLM instance.
def build_llm(model: str, **engine_kwargs):
    _require_vllm()
    return LLM(model, **engine_kwargs)


# Generate text with LLM and sampling params.
def sample(llm, prompts: list[str] | str, n: int = 1, **sample_kwargs) -> list[list[str]]:
    # Create a sampling params object
    sampling_params = SamplingParams(n=n, **sample_kwargs)

    # Generate texts from the prompt. The output is a list of RequestOutput objects
    outputs = llm.generate(prompts, sampling_params)

    # Process the outputs
    all_generated_texts = []

    for output in outputs:
        generated_texts = [output.outputs[i].text for i in range(n)]
        all_generated_texts.append(generated_texts)

    return all_generated_texts
