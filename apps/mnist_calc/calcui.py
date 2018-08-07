#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# Heather McCabe
# cv2 calculator UI elements

import cv2


class UIElement:
    def __init__(self, x, y, canvas, width=1, height=1, color=(0, 0, 0), thickness=1):
        self.x = x
        self.y = y
        self.target = canvas

        # width and height public properties defined below
        self._width = width
        self._height = height

        self.color = color
        self.thickness = thickness

    def contains_point(self, x, y):
        """Return True if a given point is within this element's boundaries, otherwise False."""
        if self.left <= x <= self.right and self.top <= y <= self.bottom:
            return True
        else:
            return False

    def clear(self):
        """Draw a filled white rectangle over this element to clear it from the canvas."""
        padding = self.thickness  # need to overwrite a slightly larger area or some drawn edges probably will remain
        cv2.rectangle(self.target, (self.left - padding, self.top - padding),
                      (self.right + padding, self.bottom + padding),
                      (255, 255, 255), cv2.FILLED)

    def draw(self):
        """Classes that inherit from UIElement should override this method appropriately."""
        raise AttributeError(str(self) + ' - draw method not implemented.')

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def center_x(self):
        return self.x + int(self.width / 2)

    @property
    def center_y(self):
        return self.y + int(self.height / 2)


class Operand(UIElement):
    def draw(self):
        # Unfilled rectangle
        cv2.rectangle(self.target, (self.left, self.top), (self.right, self.bottom), self.color, self.thickness)


class PlusSign(UIElement):
    def draw(self):
        # Horizontal line
        cv2.line(self.target, (self.center_x, self.top), (self.center_x, self.bottom),
                 self.color, self.thickness)
        # Vertical line
        cv2.line(self.target, (self.left, self.center_y), (self.left + self.width, self.center_y),
                 self.color, self.thickness)


class MinusSign(UIElement):
    def draw(self):
        # Horizontal line
        cv2.line(self.target, (self.left, self.center_y), (self.right, self.center_y), self.color, self.thickness)


class MultiplicationSign(UIElement):
    def draw(self):
        # Crossed diagonal lines (X)
        cv2.line(self.target, (self.left, self.top), (self.right, self.bottom), self.color, self.thickness)
        cv2.line(self.target, (self.left, self.bottom), (self.right, self.top), self.color, self.thickness)


class DivisionSign(UIElement):
    def draw(self):
        # Horizontal line with dots above and below
        cv2.line(self.target, (self.left, self.center_y), (self.right, self.center_y), self.color, self.thickness)
        cv2.circle(self.target, (self.center_x, self.center_y - 30), 5, self.color, 5)
        cv2.circle(self.target, (self.center_x, self.center_y + 30), 5, self.color, 5)


class EqualsSign(UIElement):
    def draw(self):
        # Top line
        cv2.line(self.target, (self.left, self.center_y - 10), (self.center_x, self.center_y - 10),
                 self.color, self.thickness)
        # Bottom line
        cv2.line(self.target, (self.left, self.center_y + 10),(self.center_x, self.center_y + 10),
                 self.color, self.thickness)


class Label(UIElement):
    def __init__(self, label, x, y, canvas, color=(0, 0, 0), thickness=1, scale=1, font=cv2.FONT_HERSHEY_DUPLEX):
        super(Label, self).__init__(x, y, canvas, color=color, thickness=thickness)
        self.label = label
        self.font = font
        self.color = color
        self.scale = scale

    def clear(self):
        """Draw a filled white rectangle over this element to clear it from the canvas."""
        padding = self.thickness  # need to overwrite a slightly larger area or some drawn edges probably will remain
        cv2.rectangle(self.target, (self.left - padding, self.top - padding),
                      (self.right + padding, self.bottom + padding * 4),  # clear more beneath to handle hanging letters like 'g'
                      (255, 255, 255), cv2.FILLED)

    def draw(self):
        # Write text label
        cv2.putText(self.target, self.label, (self.left, self.bottom), self.font, self.scale, self.color, self.thickness,
                    lineType=cv2.LINE_AA)

    @property
    def width(self):
        """Width depends on the length of the label string."""
        size, baseline = cv2.getTextSize(self.label, self.font, self.scale, self.thickness)
        return size[0]

    @property
    def height(self):
        """Height depends on the characters in the label string."""
        size, baseline = cv2.getTextSize(self.label, self.font, self.scale, self.thickness)
        return size[1]
