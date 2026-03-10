
class ConversationMemory:

    def __init__(self):
        self.history = []

    def add(self, role, message):
        self.history.append((role, message))

        if len(self.history) > 10:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(
            [f"{r}:{m}" for r, m in self.history]
        )
