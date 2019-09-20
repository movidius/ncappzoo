
class Face_detection_result():
    '''
    Used to capture the person's cropped face and bounding box coordinates when running inferences from the face_detection_retail_0004 network.
    
    Parameters
    ----------
    face_mat (OpenCV Mat) - OpenCV Mat of the person's face.
    box_left (int) - Left side of the bounding box for the face.
    box_right (int) - Right side of the bounding box for the face.
    box_top (int) - Top side of the bounding box for the face.
    box_bottom (int) - Bottom side of the bounding box for the face.
    '''
    def __init__(self, face_mat=None, box_left=None, box_right=None, box_top=None, box_bottom=None, match=False, feature_vector=None):
        self.face_mat = face_mat
        self.box_left = box_left
        self.box_right = box_right
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.match = match
        self.feature_vector = feature_vector
