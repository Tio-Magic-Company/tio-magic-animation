"""This file defines exception hierarchy for TioMagic Animation
"""

class TioMagicError(Exception):
    pass

class UnknownModelError(TioMagicError):
    # model string can't be found in the registry
    pass

class UnknownProviderError(TioMagicError):
    # provider string can't be found
    pass

class JobTimeoutError(TioMagicError):
    # Job.wait() exceeds time limit before the job finishes
    pass

class JobExecutionError(TioMagicError):
    # provider reports that job has explicitly failed
    pass

class AuthenticationError(TioMagicError):
    # missing or invalid API key or token
    pass

class APIError(TioMagicError):
    # network-related issue, like a non-200 response or JSON parsing failure
    pass

class DeploymentError(TioMagicError):
    # target infrastructure is unavailable (e.g. Modal app name is wrong)
    pass

class ValidationError(TioMagicError):
    # user provided invalid input parameters
    pass

class RateLimitError(TioMagicError):
    # provider's rate limit has exceeded
    pass