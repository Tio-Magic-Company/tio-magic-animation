"""Registry system for model implementations
Maintains mapping of feature/model/provider combinations to their implementations
"""

class Registry:
    """Global registry for all model implementations"""
    def __init__(self):
        self._implementations = {}
        self._features = set()
        self._models = set()
        self._providers = set()
    def register(self, feature, model, provider, implementation):
        """Register an implementation for a feature/model/provider combo
        
        Args:
            feature (str): Feature name (e.g., 'text_to_video')
            model (str): Model name (e.g., 'wan-2.1')
            provider (str): Provider name (e.g., 'modal')
            implementation: Implementation class
        """
        key = (feature, model, provider)
        # Update our sets for quick querying
        self._features.add(feature)
        self._models.add(model)
        self._providers.add(provider)

        # Register the implementation
        self._implementations[key] = implementation
    def get_implementation(self, feature, model, provider):
        """Get the implementation for a specific combination
        
        Args:
            feature (str): Feature name
            model (str): Model name
            provider (str): Provider name
            
        Returns:
            Implementation class
            
        Raises:
            ValueError: If no implementation exists for the combination
        """
        key = (feature, model, provider)
        if key not in self._implementations:
            # Try with default model if none specified
            if model is None:
                # Find the first available model for this feature/provider
                for (f, m, p), impl in self._implementations.items():
                    if f == feature and p == provider:
                        return impl

            # If still not found
            raise ValueError(f"No implementation found for {feature}/{model}/{provider}")

        return self._implementations[key]
    def get_features(self):
        """Get all supported features
        
        Returns:
            list: Sorted list of supported features
        """
        return sorted(list(self._features))

    def get_providers(self):
        """Get all supported providers
        
        Returns:
            list: Sorted list of supported providers
        """
        return sorted(list(self._providers))

    def get_models(self, feature=None, provider=None):
        """Get all available models, optionally filtered by feature/provider
        
        Args:
            feature (str, optional): Filter models by feature
            provider (str, optional): Filter models by provider
            
        Returns:
            list: Sorted list of model names
        """
        models = set()
        for (feat, model, prov) in self._implementations.keys():
            if (feature is None or feat == feature) and (provider is None or prov == provider):
                models.add(model)
        return sorted(list(models))

    def get_supported_providers(self, feature, model):
        """Get providers that support a specific feature/model combination
        
        Args:
            feature (str): Feature name
            model (str): Model name
            
        Returns:
            list: Providers supporting this feature/model
        """
        providers = []
        for (feat, mod, prov) in self._implementations.keys():
            if feat == feature and mod == model:
                providers.append(prov)
        return sorted(providers)

    def get_supported_features(self, model, provider):
        """Get features supported by a specific model/provider combination
        
        Args:
            model (str): Model name
            provider (str): Provider name
            
        Returns:
            list: Features supported by this model/provider
        """
        features = []
        for (feat, mod, prov) in self._implementations.keys():
            if mod == model and prov == provider:
                features.append(feat)
        return sorted(features)

# Create global registry instance
registry = Registry()