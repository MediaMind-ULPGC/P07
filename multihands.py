from handperson import HandPerson
import numpy as np

class MultiHandTracker:
    def __init__(self, fps):
        self.tracked_hands = []
        self.fps = fps

    def update(self, keypoints):
        for keypoint in keypoints:
            keypoint_coordinates = keypoint.xy[0]
            right_wrist = keypoint_coordinates[10]
            conf = keypoint.conf[0, 10]
            if conf > 0.2:
                x, y = int(right_wrist[0]), int(right_wrist[1])
                self.add_or_update_hand(x, y)
    
    def add_or_update_hand(self, x, y, threshold=30):
        for hand in self.tracked_hands:
            last_x, last_y = hand.x[-1], hand.y[-1]
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if distance < threshold:
                hand.calculate_distance(x, y)
                return
        
        new_hand = HandPerson(fps=self.fps)
        new_hand.add(x, y)
        self.tracked_hands.append(new_hand)