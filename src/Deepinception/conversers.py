from langchain_ollama import ChatOllama


def load_models(provider: str, model: str, kwargs: dict):
    """
    Factory to create target model wrapper using LangChain-based GPT wrapper.
    """
    num_predict = kwargs.pop("target_max_n_tokens", 256)
    temperature = kwargs.pop("target_temperature", 0.0)
    if provider == "Ollama":
        model = ChatOllama(model=model, num_predict=num_predict, temperature=temperature, **kwargs)
    else:
        raise NotImplementedError(f"Provider {provider} not implemented.")
    return model
