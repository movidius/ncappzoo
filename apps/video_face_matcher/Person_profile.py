
class Person_profile():
    '''
    Used to identify the person with an image read from file. Used when identifying input and output that will be sent/received with the face_detection_retail_0004 network.
    
    Parameters
    ----------
    name (str) - Name of the person.
    image_path (str) - File path of the image of the person.
    image_mat (OpenCV Mat) - OpenCV Mat of the image of the person.
    '''
    def __init__(self, name=None, image_path=None, image_mat=None):
        self.name = name
        self.image_path = image_path
        self.image_mat = image_mat


