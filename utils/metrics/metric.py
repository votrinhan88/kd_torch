from abc import ABC, abstractmethod

class Metric(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}(latest={self.latest:.4g})"

    """Base class for metrics."""
    @abstractmethod
    def update(self):
        """Update metric from given inputs."""        
        pass

    @abstractmethod
    def reset(self):
        """Reset tracked parameters, typically used when moving to a new epoch."""        
        pass

    @property
    def latest(self):
        """Return the latest metric value in a pythonic format."""
        return self.value.item()