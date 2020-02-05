from servo import Servo
import pigpio

class MeArmPi:
    def __init__(self):
        self.servos = {}
        self.sleeper = None
        pi = pigpio.pi()
        self.servos['base'] = Servo({'pin': 4, 'min': 530, 'max': 2400, 'minAngle': -90, 'maxAngle': 90}, pi)
        self.servos['lower'] = Servo({'pin': 17, 'min': 1300, 'max': 2400, 'minAngle': 0, 'maxAngle': 135}, pi)
        self.servos['upper'] = Servo({'pin': 22, 'min': 530, 'max': 2000, 'minAngle': 0, 'maxAngle': 135}, pi)
        self.servos['grip'] = Servo({'pin': 10, 'min': 1400, 'max': 2400, 'minAngle': 0, 'maxAngle': 90}, pi)
        self.moveToCenters()

    def moveToCenters(self):
        for s in self.servos.keys():
            self.servos[s].moveToCenter()

    def moveBaseTo(self, angle):
        return self.servos['base'].moveToAngle(angle)

    def moveLowerTo(self, angle):
        return self.servos['lower'].moveToAngle(angle)

    def moveUpperTo(self, angle):
        return self.servos['upper'].moveToAngle(angle)

    def moveGripTo(self, angle):
        return self.servos['grip'].moveToAngle(angle)

    def openGrip(self):
        return self.moveGripTo(0)

    def closeGrip(self):
        return self.moveGripTo(90)

    def moveByPercent(self, servo, percent):
        return self.servos[servo].moveByPercent(percent)
    
    def moveServoTo(self, servo, angle):
        return self.servos[servo].moveToAngle(angle)
