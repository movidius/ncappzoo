import pigpio
import time
from math import floor

class Servo:
    def __init__(self, conf, pi):
        self.conf = conf
        self.currentAngle = 0
        self.releaseTimeout = None
        self.lastPulseWidth = 0
        self.sleeping = False
        self.pi = pi
        pi.set_mode(conf['pin'], pigpio.OUTPUT)

    def moveToAngle(self, angle):
        return self.setCurrentAngle(angle)

    def sleep(self):
        self.pi.write(self.conf['pin'], 0)
        self.sleeping = True

    def wake(self):
        if self.sleeping:
            self.updateServo();
            self.sleeping = False

    def updateServo(self):
        newPulseWidth = self.calculatePulseWidth(self.currentAngle)
        diff = abs(self.lastPulseWidth - newPulseWidth)
        self.lastPulseWidth = newPulseWidth
        duration = (diff / 1900) * 500 / 1000 # still wondering what to do...
        self.pi.set_servo_pulsewidth(self.conf['pin'], newPulseWidth)
        time.sleep(duration)
        print("set " + str(self.conf['pin']))
        return True

    def setCurrentAngle(self, angle):
        angle = int(angle)
        if angle < self.conf['minAngle']:
            self.currentAngle = self.conf['minAngle']
        elif angle > self.conf['maxAngle']:
            self.currentAngle = self.conf['maxAngle']
        else:
            self.currentAngle = angle
        status = self.updateServo()
        return status

    def moveByDegrees(self, angle):
        return self.setCurrentAngle(self.currentAngle + angle)

    def moveByPercent(self, percent):
        part = self.conf['maxAngle'] - self.conf['minAngle']
        part = part / 100
        part = part * percent
        return self.setCurrentAngle(self.currentAngle + part)

    def calculatePulseWidth(self, angle):
        diff_w_min = angle - self.conf['minAngle']
        diff_w_maxmin_ang = self.conf['maxAngle'] - self.conf['minAngle']
        diff_w_maxmin = self.conf['max'] - self.conf['min']
        divide = diff_w_min / diff_w_maxmin_ang
        return floor(self.conf['max'] - (divide * diff_w_maxmin))

    def moveToCenter(self):
        diff_w_maxmin_ang = self.conf['maxAngle'] - self.conf['minAngle']
        center = floor(self.conf['minAngle'] + diff_w_maxmin_ang / 2)
        return self.moveToAngle(center)
