#Import libraries
################################################
import matplotlib.pyplot as plt
import numpy as np
import random 

#Generate dataset
################################################

np.random.seed(1) # Set seed for reproducing
n = 10 # number of samples
w = np.random.uniform(n,13.5)
beta = np.random.uniform(30,40)
x = np.linspace(0,4, n) 
y = w*x+beta + np.random.uniform(-5, 5, n)
print(m,c) # Print true relationship values

# Analytic solution (linear least squares)
################################################
x = np.array([np.ones(n),x]).T
W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),
                       np.transpose(x)),y)
beta = W[0]
omega = W[1]
print(beta, omega)

# Define linear and loss functions 
################################################
def linear_node(x, weight, bias):
  z = weight*x + bias
  return z

# Loss function (mean squared distance metric)
def squared_loss(y_exact, y_approx):
  residual = y_exact-y_approx
  return residual**2

def dY_squared_loss(y_exact, y_approx):
  return -2*(y_exact - y_approx)

# Numeric loop; Gradient descent and Newton Raphson
################################################

# Number of iterations
epochs = 100

# Variables for gradient descent
nu = 0.005 # MAKE SURE THIS IS SMALLER OTHERWISE DIVERGENCE!!!!
bias_gd = np.random.uniform(-0.5, 0.5)
weight_gd = np.random.uniform(-0.5, 0.5)

# Keep track of loss, weight, and bias at for each iterations (Gradient Descent)
loss_gd = []
weight_values_gd = [bias_gd]
bias_values_gd = [weight_gd]


# Variables for Newton-Raphson
bias_nr = np.random.uniform(-0.5, 0.5)
weight_nr = np.random.uniform(-0.5, 0.5)
weights = np.array([bias_nr,weight_nr]).T

# Keep track of loss, weight, and bias at for each iterations (Newton-Raphson)
loss_nr = []
weight_values_nr = [weight_nr]
bias_values_nr = [bias_nr]

conv_gd = 0
conv_nr = 0
# Main loop
for i in range(epochs):
  # Calculate Linear combination
  y_gd_approx = linear_node(x, weight_gd, bias_gd)
  y_nr_approx = linear_node(x, weight_nr, bias_nr)

  inst_loss_gr = sum(squared_loss(y,y_gd_approx))
  inst_loss_nr = sum(squared_loss(y,y_nr_approx))

  # calculate losses
  loss_gd.append(inst_loss_gr)
  loss_nr.append(inst_loss_nr)

  # Update Scheme for gradient descent
  dy_loss_gd = dY_squared_loss(y, y_gd_approx)
  weight_gd = weight_gd - nu*np.dot(dy_loss_gd.T, x)
  bias_gd = bias_gd - nu*np.sum(dy_loss_gd)

  # Append updated weight and bias to tracking list
  weight_values_gd.append(weight_gd)
  bias_values_gd.append(bias_gd)

  # Update Scheme for Newton-Raphson
  dy_loss_nr = dY_squared_loss(y, y_nr_approx)
  score   = np.array([np.sum(dy_loss_nr), np.dot(dy_loss_nr.T, x)]).T
  Hessian = np.array([[2*n, -2*n*weight_nr], [2*sum(x), sum(-dy_loss_nr-weight_nr*x)]]).T
  weights = weights - np.dot(np.linalg.inv(Hessian),score)

  bias_nr = weights[0]
  weight_nr = weights[1]

  # Append updated weight and bias to tracking list
  weight_values_nr.append(weight_nr)
  bias_values_nr.append(bias_nr)

for i in range(0,len(loss_nr)-1):
  if (abs(loss_nr[i+1] - loss_nr[i]) < 1):
    conv_nr = i+1
    break
for i in range(0,len(loss_gd)-1):
  if (abs(loss_gd[i+1] - loss_gd[i]) < 1):
    conv_gd = i+1
    break

print(weight_gd, bias_gd)

# Create plots
################################################
plt.style.use('seaborn')
fig, ax = plt.subplots(3,1)
# Plot loss per epoch of gradient descent
ax[0].plot(np.arange(0, epochs, 1), loss_gd, color='red', label='GD')
# Plot loss per epoch of newton-raphson
ax[0].plot(np.arange(0, epochs, 1), loss_nr, color='green', label='Nr')
# Plot optimal minimum loss (LLS)
ax[0].plot(np.arange(0, epochs, 1), loss_analytic*np.ones(epochs), color='black', 
        label='LLS', ls='--',)
# Plot point of convergence 
ax[0].vlines(x=conv_gd, ymin=loss_analytic-20, ymax=loss_analytic+75, colors='red', ls=':', lw=2, label='GD convergence')
ax[0].vlines(x=conv_nr, ymin=loss_analytic-20, ymax=loss_analytic+75, colors='green', ls=':', lw=2, label='NR convergence')
# Set axis labels, title
ax[0].set(xlabel='Epochs', ylabel='Loss')
ax[0].legend(loc='upper right')
ax[0].set_title("Loss v Epochs")
ax[0].xlim([-0.5, epochs])
ax[0].ylim([0,1000])

ax[1].xaxis.get_major_locator().set_params(integer=True)
# Plot loss per iteration of gradient descent
ax[1].plot(np.arange(0, epochs+1, 1), weight_values_gd, color='C3', label='GD: Weight')
ax[1].plot(np.arange(0, epochs+1, 1), bias_values_gd, color='C0', label='GD: Bias')
# Plot loss per iteration of Netwon-Raphson
ax[1].plot(np.arange(0, epochs+1, 1), weight_values_nr, color='C5', label='NR: Weight')
ax[1].plot(np.arange(0, epochs+1, 1), bias_values_nr, color='C1', label='NR: Bias')
# Plot loss per iteration of Analytic
ax[1].plot(np.arange(0, epochs, 1), beta_a[1]*np.ones(epochs), color='C8', 
        label='A: Weight', ls='--')
ax[1].plot(np.arange(0, epochs, 1), beta_a[0]*np.ones(epochs), color='C9', 
        label='A: Bias', ls='--')
#Set xlim, ylim (plot range)
ax[1].set(xlabel='Iterations', ylabel='Weight')
ax[1].legend(loc='upper right')
ax[1].set_title("Weights v Epochs")

y_analytic = beta_a[1]*x + beta_a[0]
y_GD = weight_gd*x + bias_gd
y_NR = weight_nr*x + bias_nr

# Draw points
ax[2].scatter(y, x, color='#71A8CF', label='Data')
# Plot Analytic solution
ax[2].plot(y_analytic, x, color='red', label='Analytic')
#Exact solution
ax[2].plot(m*x+c, x, color='blue', label='Exact', ls='--')
# Plot GD solution
ax[2].plot(y_GD, x, color='green', label='GD')
# Plot NR solution
ax[2].plot(y_NR, x, color='orange', label='NR')
# Set captions
ax[2].xlabel('Grade')
ax[2].ylabel('Hours')
ax[2].title('Statistics grade V Hours studying')
#Set xlim, ylim (plot range)
ax[2].xlim([25, 110])
ax[2].ylim([-0.25, 4.2])
ax[2].legend(loc='upper left')


fig.tight_layout()
plt.show()

# Plot loss v iterations
################################################

