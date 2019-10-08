          
class My_network():
    '''
    Used to hold various network parameters.
    
    Parameters
    ----------
    exec_net (OpenVINO ExecutableNetwork) - OpenVINO ExecutableNetwork Object.
    input_node_name (str) - The network's input node/layer name.
    output_node_name (str) - The network's output node/layer name.
    input_w (int) - The network's input shape width.
    input_h (int) - The network's input shape height.
    
    '''
    def __init__(self, exec_net=None, input_node_name=None, output_node_name=None, input_w=None, input_h=None):
         self.exec_net = exec_net
         self.input_node_name = input_node_name
         self.output_node_name = output_node_name
         self.input_w = input_w
         self.input_h = input_h
   
