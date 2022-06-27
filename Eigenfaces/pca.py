'''
Created on Mar 7, 2020

@author: Admin
'''
import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

'''load the dataset from a provided .mat file, 
re-center it around the origin and return it as a NumPy array of floats
'''
def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = np.array(dataset['fea'])
    x = x - np.mean(x,axis = 0)
    return x

'''calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
'''
def get_covariance(dataset):
    arr = np.array(dataset)
    arr = np.dot(np.transpose(arr),arr)
    arr = arr/(len(dataset)-1)
    return arr

'''perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array)
with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding 
eigenvectors as columns
'''
def get_eig(S,m):
    eigenValues ,eigenVectors = eigh(S,eigvals=(len(S)-m,len(S)-1))
    tmp = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[tmp]
    eigenValues = np.diag(eigenValues)
    eigenVectors = eigenVectors[:,tmp]
    return eigenValues, eigenVectors

''' project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
'''
def project_image(image,U):
    return np.dot(np.dot(image,U),np.transpose(U))

'''use matplotlib to display a visual representation of the original image and the projected image 
side-by-side
'''
def display_image(orig,proj):
    orig = np.reshape(orig,(32,32))
    orig = np.transpose(orig)
    proj = np.reshape(proj,(32,32))
    proj = np.transpose(proj)
    fig,axs =plt.subplots( figsize = (10,3), ncols = 2)
    axs[0].set_title('Original')
    a =axs[0].imshow(orig,aspect='equal')
    fig.colorbar(a, ax=axs[0])
    axs[1].set_title('Projection')
    b =axs[1].imshow(proj,aspect='equal')
    fig.colorbar(b, ax=axs[1])
    plt.show()
