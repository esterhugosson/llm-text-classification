# Allowed interactions return true

class InteractionFilter:

    def allow(self, interaction):
        if interaction.role != self.role:
            return False
        if len(interaction.text) < 10:
            return False
        return True