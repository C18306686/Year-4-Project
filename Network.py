import random
import numpy as np
import Value # Must have Value class in same localtion as Network
def loss(y,hat_y): 
  return (y-hat_y)**2

def log_loss(y_exact, y_approx):
  return -(y_exact.data*math.log(y_approx.data)+(1-y_exact.data)*math.log(1-y_approx.data))

class Network:
  def __init__(self, activation='relu'):
    self.activation = self.relu if activation is 'relu' else self.sigmoid
    self.precompiled_layers = []
    self.layers = []
    self.bias = []
    self.parameters = np.array([])
    self.num_layers = 0
    self.input_dim = 0
    self.output_dim = 0
    self.input_layer = False
    self.output_layer = False
    self.compiled = False

  def add_layer(self, num_nodes):
    if self.output_layer:
      print("Network has output layer, cannot add new layers!")
      return 0 
    self.num_layers+=1   
    self.precompiled_layers.append(num_nodes)
    return 1
  def add_input(self, num_features):
    if self.compiled:
      print("Network already compiled!")
      return 0
    self.input_dim = num_features
    self.num_layers+=1   
    if self.input_layer:
      self.precompiled_layers[0] = num_features
    else:
      self.precompiled_layers.insert(0, num_features)
    self.input_layer = True
    return 1
  def add_output(self, num_targets):
    if self.compiled:
      print("Network already compiled!")
      return 0
    if self.output_layer:
      print("Network already has output layer!")
      return 0
    self.output_layer = True
    self.output_dim = num_targets
    self.precompiled_layers.append(num_targets)
    return 1

  def compile(self):
    if self.compiled:
      print("Network already compiled!")
      return 0
    if not self.input_layer:
      print("Network has no input layer!")
      return 0
    if not self.output_layer:
      print("Network has no output layer!")
      return 0
    for indx in range(1, len(self.precompiled_layers)):
      # Create matrices for each layer, row_i = (node input)_i
      self.layers.append(np.array([np.array([Value(random.random()) for node_input in range(self.precompiled_layers[indx-1])]) for nodes in range(self.precompiled_layers[indx])]))
      # Create column vectors for bias terms for each layer
      self.bias.append(np.array([Value(random.random()) for node in range(self.precompiled_layers[indx])]).T)
    print(self.layers)
    for i in self.layers: 
      self.parameters = np.append(self.parameters, i.flatten())
    for i in self.bias:
      self.parameters = np.append(self.parameters, i.flatten())
    self.parameters = self.parameters.flatten()

  def __call__(self,x):
    if isinstance(x,(int,float)):
      x = [x]
    if len(x) != self.input_dim:
      print(f"Incorrect input dimension. Expected {self.input_dim}, Given {len(x)}")
      return 0
    output = self.activation(self.layers[0], x, self.bias[0])
    for i in range(1,self.num_layers-1):
      output = self.activation(self.layers[i], output, self.bias[i])
    return self.linear(self.layers[-1], output, self.bias[-1]) # Change relu to sigmoid!
  def zero_grad(self):
    for p in self.parameters:
      p.grad=0
  # Define activation functions
  def relu(self, w,x,b):
    """A single neuron computing either a linear function of inputs or a ReLU applied to linear function"""
    v =  self.linear(w,x,b)# Should return column vecotr
    v = v.T                # Transpose to row vector
    return np.array([i.relu() for i in v]).T
  def sigmoid(self, w,x,b):
    """A single neuron computing either a linear function of inputs or a ReLU applied to linear function"""
    v =  self.linear(w,x,b)# Should return column vecotr
    v = v.T                # Transpose to row vector
    return np.array([i.sigmoid() for i in v]).T
  def linear(self, w,x,b):
    z =  np.matmul(w,x)
    return z+b

  def node_outputs(self,x):
    layer_1 = [self.activation(val, x, self.bias[iter]) for iter,val in enumerate(self.input_layer)]
    all_layers = [*[i.data for i in layer_1],self.linear(self.output, layer_1, self.bias[-1]).data]
    return all_layers
