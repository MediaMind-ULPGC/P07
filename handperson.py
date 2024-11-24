import numpy as np

class HandPerson():

    def __init__(self):
        self.x = []
        self.y = []
    
    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
    
    def calculate_distance(self, x, y):
        if len(self.x)>0 and len(self.y)>0:
            distance = np.sqrt((x-self.x[-1])**2 + (y-self.y[-1])**2)
            if distance<5:
                self.add(x,y)
            else:
                return None
        else:
            self.add(x,y)