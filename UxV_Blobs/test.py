class Fish:
    def __init__(self, name='Fish'):
        self.name = name

class Green(Fish):
    def __init__(self, name='GreenFish'):
        super().__init__(name)

fish = Fish


greenfish = Green