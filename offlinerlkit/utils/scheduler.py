class LinearParameter:
    def __init__(self, start=1.0, end=0.1, num_steps=10):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_decrement = (start - end) / float(num_steps)
        self.value = start
        self.current_step = 0

    def decrease(self):
        """Decreases the parameter linearly for one step and ensures it doesn't go below the minimum."""
        if self.current_step < self.num_steps:
            self.value -= self.step_decrement
            self.value = max(self.value, self.end)
            self.current_step += 1