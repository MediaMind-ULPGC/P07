import matplotlib.pyplot as plt
import numpy as np

class HandPerson():

    def __init__(self, fps):
        self.x = []
        self.y = []
        self.speeds = []
    
    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
        if len(self.x) > 1:
            dx = x - self.x[-2]
            dy = y - self.y[-2]
            distance = np.sqrt(dx**2 + dy**2)
            self.speeds.append(distance)
    
    def calculate_distance(self, x, y, threshold=30):
        if len(self.x) > 0 and len(self.y) > 0:
            distance = np.sqrt((x - self.x[-1])**2 + (y - self.y[-1])**2)
            if distance < threshold:
                self.add(x, y)
            else:
                return None
        else:
            self.add(x, y)