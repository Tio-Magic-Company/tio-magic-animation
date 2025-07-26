"""Configuration management for TioMagic
Handles provider settings, API keys, and other global confiurations
"""

from tiomagic.core.errors import UnknownProviderError


class Configuration:
    def __init__(self):
        self._provider = "modal" #default
        self._api_keys = {}
        self._model_path = None #local models
        self._options = {} #additional provider-specific options
    def get_provider(self):
        return self._provider
    def set_provider(self, provider):
        supported_providers = ["modal", "local", "baseten"]
        if provider not in supported_providers:
            supported_providers = ', '.join(supported_providers)
            raise UnknownProviderError(provider=provider, available_providers=supported_providers)

        self._provider = provider

    def get_api_key(self, provider=None):
        if provider is None:
            provider = self._provider # use active provider
        return self._api_keys.get(provider)
    def set_api_key(self, provider, key):
        if key is not None:
            self._api_keys[provider] = key

    def get_model_path(self):
        return self._model_path
    def set_model_path(self, path):
        self._model_path = path

    def get_option(self, key, default=None):
        return self._options.get(key, default)
    def set_option(self, key, value):
        self._options[key] = value
    def get_all_options(self):
        return self._options.copy()

