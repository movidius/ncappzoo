#! /usr/bin/env python3

# Copyright(c) 2020 Intel Corporation.
# License: MIT See LICENSE file in root directory.

import brickpi3
import time
import logging as log


class DriveBrickPi3:
    """
    This class for controlling (car BrickPi3 lego car)
    """

    def __init__(self):
        """
        Initialize brickpi3 and set name, speed and status(direction)
        """
        self.BP = brickpi3.BrickPi3()
        self.BP.offset_motor_encoder(self.BP.PORT_A, self.BP.get_motor_encoder(self.BP.PORT_A))
        self.BP.offset_motor_encoder(self.BP.PORT_A, self.BP.get_motor_encoder(self.BP.PORT_D))
        self.BP.set_motor_limits(self.BP.PORT_A, 50, 200)
        self.BP.set_motor_limits(self.BP.PORT_D, 50, 200)
        self.name = "IntelCar"
        self.speed = 18        
        self.status = CAR_DIRECTION.FWD
        log.info("Car {} has been started with speed: {}".format(str(self.name), str(self.speed)))
        time.sleep(0.02) # delay for 0.02 seconds (20ms) to reduce the Raspberry Pi CPU load.

    def __str__(self):
        """
        Print string
        :return: car data
        """
        return ("Encoder A: %4d  D: %4d  Speed:  %d  Name: %d" %
                (self.BP.get_motor_encoder(self.BP.PORT_A),
                 self.BP.get_motor_encoder(self.BP.PORT_D),
                 self.speed, self.name))

    @property
    def speed(self):
        """
        Get speed value of the car
        :return: speed value
        """
        return self.__speed

    @speed.setter
    def speed(self, val):
        """
        Set speed value for the car
        :param val: |int| speed number 0-100
        """
        self.__speed = val

    @property
    def status(self):
        """
        Get status of the car (mostly directions)
        :return: status value
        """
        return self.__status

    @status.setter
    def status(self, val):
        """
        Set status for the car
        :param val: |Enum| CAR_DIRECTION - directions
        """
        self.__status = val

    @property
    def name(self):
        """
        Get name value of the car
        :return: name value
        """
        return self.__name

    @name.setter
    def name(self, val):
        """
        Set name value for the car
        :param val: |String| car name value
        """
        self.__name = val

    def reset(self):
        """
        Unconfigure the sensors, disable the motors, and restore the LED
        to the control of the BrickPi3 firmare
        """
        self.BP.reset_all()
        
    def u_turn(self):
        """
        Reset all functions of brickpi
        """
        self.BP.set_motor_position(self.BP.PORT_A, 180)

    def move_car(self, direction):
        """
        Move car by declaring the direction where to go
        :param direction: |Enum| CAR_DIRECTION - directions
        """
        if direction is CAR_DIRECTION.FWD:
            self.BP.set_motor_power(self.BP.PORT_A + self.BP.PORT_D, self.speed)

        elif direction is CAR_DIRECTION.REVERSE:
            self.BP.set_motor_power(self.BP.PORT_A + self.BP.PORT_D, -self.speed)

        elif direction is CAR_DIRECTION.STOP:
            self.BP.set_motor_power(self.BP.PORT_A + self.BP.PORT_D, 0)

        elif direction is CAR_DIRECTION.RIGHT:
            self.BP.set_motor_power(self.BP.PORT_A, -self.speed / 1.5)
            self.BP.set_motor_power(self.BP.PORT_D, self.speed)

        elif direction is CAR_DIRECTION.LEFT:
            self.BP.set_motor_power(self.BP.PORT_A, self.speed)
            self.BP.set_motor_power(self.BP.PORT_D, -self.speed / 1.5)


class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


# Get car direction #
CAR_DIRECTION = Enum(["FWD", "REVERSE", "RIGHT", "LEFT", "STOP"])
