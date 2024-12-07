import matplotlib.pyplot as plt
import numpy as np

class HandPerson():

    def __init__(self, fps):
        self.x = []
        self.x_skip = []
        self.y = []
        self.y_skip = []
        self.speeds = []
        self.time_frames = []
    
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
                self.add_skip(x, y)
                return True
            else:
                return False
        else:
            self.add(x, y)
            self.add_skip(x, y)
            return True
    
    def add_frame(self, frame):
        self.time_frames.append(frame)
    
    def add_skip(self, x, y):
        self.x_skip.append(x)
        self.y_skip.append(y)