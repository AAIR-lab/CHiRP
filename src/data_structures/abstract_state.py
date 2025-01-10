#TODO

class AbstractState(object):

    def __init__(self, state, identifier):
        self.state = state
        self.state_tuple = None
        self.id = identifier
        self.concrete_states = set()

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return str([[round(float(j),2) for j in i.split(",")] for i in self.state]) + "_" + str(self.id)

    def __repr__(self) -> str:
        return str(self.state) + "_" + str(self.id)

    def __eq__(self, other):
        return self.__hash__() == other.__hash()

    def add_concrete_state(self, concrete_state):
        self.concrete_states.add(tuple(concrete_state))