from abc import ABC, abstractmethod



class BesiegeStrategy(ABC):
    @abstractmethod
    def besiege(self, ctx):
        pass
