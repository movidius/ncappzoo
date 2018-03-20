import cv2


def detect(img):
    """Get a list of boundingRects for objects detected in an image."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [cv2.boundingRect(contour) for contour in contours]


def sort(rects):
    """Sort a list of boundingRects into sublists by vertical row.
    The sublists will also be sorted left to right horizontally.
    """
    rows = []

    # First sort rects by the y-coordinate from lowest to highest (top of the image to the bottom)
    rects = sorted(rects, key=lambda r: r[1])

    # Split the rects into rows
    y_coord = 0
    sublist = []
    for rect in rects:
        x, y, w, h = rect

        if y > y_coord:
            y_coord = y + h

            if sublist:
                # Start a new row
                rows.append(sublist)
                sublist = []

        sublist.append(rect)
    rows.append(sublist)

    # Sort the sublists of rects by the x-coordinate from lowest to highest (the left of the image to the right)
    rows = [sorted(row, key=lambda r: r[0]) for row in rows]

    return rows