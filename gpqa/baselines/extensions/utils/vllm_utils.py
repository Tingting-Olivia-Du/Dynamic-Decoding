from vllm import LLM, SamplingParams


# Create an offline LLM instance.
def build_llm(model: str, **engine_kwargs):
    llm = LLM(model, **engine_kwargs)
    return llm


# Generate text with LLM and sampling params.
def sample(llm: LLM, prompts: list[str] | str, n: int = 1, **sample_kwargs) -> list[list[str]]:
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
