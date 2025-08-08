"""Configuration management for TioMagic
Handles provider settings, API keys, and other global confiurations
"""

from .errors import UnknownProviderError


class Configuration:
    def __init__(self):
        self._provider = "modal" #default
        self._api_keys = {}
        self._model_path = None #local models
        self._modal_options = {
            "gpu": None,
            "timeout": None,
            "scaledown_window": None,
        }
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
    
    def set_gpu(self, gpu):
        self._gpu = gpu
    def set_timeout(self, timeout):
        self._timeout = timeout
    def set_scaledown(self, scaledown_window):
        self._scaledown_window = scaledown_window

    def set_modal_options(self, gpu, timeout, scaledown_window):
        self._modal_options['gpu'] = gpu
        self._modal_options['timeout'] = timeout
        self._modal_options['scaledown_window'] = scaledown_window
    def get_modal_options(self):
        return self._modal_options
