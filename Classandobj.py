class Car:
    def __init__(self,name,color):
        self.name=name
        self.color=color
        self.speed=0
    def increase_speed(self):
        self.speed=self.speed+10
    def decrease_speed(self):
        self.speed=self.speed-10
