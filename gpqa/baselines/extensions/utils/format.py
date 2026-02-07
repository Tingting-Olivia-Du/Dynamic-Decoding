# for applying chat templates
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
        task_prompt: str,
        response_prefix: str,
        tokenizer,
        **kwargs,
) -> str:
    # check if the tokenizer has a tokenizer.chat_template method
    if tokenizer is None or tokenizer.chat_template is None:
        return task_prompt + '\n' + response_prefix

    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response_prefix + _MAGIC_SPLITTER_},
        ],
        tokenize=False,
        **kwargs,
    ).split(_MAGIC_SPLITTER_)[0]

    return task_prompt
