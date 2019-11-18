from mearm_pi import MeArmPi

class MeArmController:
    def __init__(self):
        self.mearm = MeArmPi()
        self.angles = {'base': 0, 'lower': int(135/2), 'upper': int(135/2), 'grip': 45}
        self.bounds = {'base': (-90, 90), 'lower': (0, 135), 'upper': (0, 135), 'grip': (0, 90)}
    
    def move(self, servo, gesture):
        angle = self.angles[servo]
        minBound = self.bounds[servo][0]
        maxBound = self.bounds[servo][1]
        
        if gesture == 1 and angle > minBound:
            self.angles[servo] = self.angles[servo] - 1 # up
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 2 and angle < maxBound:
            self.angles[servo] = self.angles[servo] + 1 # down
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 3 and angle > minBound:
            self.angles[servo] = self.angles[servo] - 1 # left
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 4 and angle < maxBound:
            self.angles[servo] = self.angles[servo] + 1 # right
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 5 and angle < maxBound:
            self.angles[servo] = self.angles[servo] + 1 # open
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 6 and angle > minBound:
            self.angles[servo] = self.angles[servo] - 1 # close
            return self.mearm.moveServoTo(servo, self.angles[servo])
    
        if gesture == 7 and angle > minBound:
            self.angles[servo] = self.angles[servo] - 1 # out
            return self.mearm.moveServoTo(servo, self.angles[servo])
        
        if gesture == 8 and angle < maxBound:
            self.angles[servo] = self.angles[servo] + 1 # out
            return self.mearm.moveServoTo(servo, self.angles[servo])
    