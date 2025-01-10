
class Sample():
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward


class Transition():
    def __init__(self, state, action, option, next_state, reward=None, done=None):
        self.state = state
        self.action = action
        self.option = option
        self.next_state = next_state
        self.reward = reward
        self.done = done
    
    def __str__(self) -> str:
        return f"{self.state} {self.action} {self.next_state}"


class Trajectory():
    def __init__(self):
        self.trajectory = []

    def append_transition(self, transition):
        self.trajectory.append(transition)
        