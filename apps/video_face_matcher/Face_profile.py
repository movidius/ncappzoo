
class Face_profile():
    '''
    Used to capture the person's name with results from facenet inference. The feature vector will be used to identify the person's face.
    
    Parameters
    ----------
    name (str) - Name of the person.
    image_path (str) - File path of the image of the person.
    feature_vector (List of 512 floats) - List of 512 dimensional embeddings.
    '''
    def __init__(self, name=None, image_path=None, feature_vector=None):
        self.name = name
        self.image_path = image_path
        self.feature_vector = feature_vector

