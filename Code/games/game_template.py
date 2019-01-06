from abc import ABC, abstractmethod

from copy import deepcopy


class AbstractGameInstance(ABC):

    @abstractmethod
    def do_action(self, action):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def is_terminated(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def clone(self):
        return deepcopy(self)
