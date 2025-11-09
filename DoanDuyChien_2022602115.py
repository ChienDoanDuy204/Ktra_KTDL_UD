import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_normalization(x,a,b):
  return (b-a)*(x-np.min(x))/(np.max(x)-np.min(x))+a
a = None
b = None
X = np.apply_along_axis(data_normalization,arr=X, axis = 0, a = a, b= b)
# ---------------------------------------------------------
class GaussianNavieBayes:
  def fit(self,X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    # Lấy ra các nhãn đơn nhất và số lần xuất hiện của chúng
    self.Labels_unique,count = np.unique(self.y_train,return_counts=True)
    # Tính xác xuất xảy ra của các nhãn
    self.P_labels = count/y_train.size
    # lấy log(probability)
    self.P_labels = np.log(self.P_labels)

    # Mảng mean của feature tương ứng với label = i xảy ra
    self.matrix_mean = np.zeros((self.Labels_unique.size,X_train.shape[1]))
    # Mảng mean của feature tương ứng với label = i xảy ra
    self.matrix_var = np.zeros((self.Labels_unique.size,X_train.shape[1]))

    for i in range(self.Labels_unique.size):
      data_label_i = self.X_train[self.y_train==self.Labels_unique[i]]
      self.matrix_mean[i] = np.mean(data_label_i,axis=0)
      self.matrix_var[i] = np.var(data_label_i,axis=0) + 1e-6


  def predict(self,X_test):
    if X_test.ndim == 1:
      raise ValueError("X_test must be 2D")
    else:
      self.X_test = X_test
      # Xác suất của nhãn i xảy ra khi X_test xảy ra
      self.P_labels_X_test = np.zeros(self.P_labels.shape)
      self.P_labels_X_test+=self.P_labels
      for i in range(self.Labels_unique.size):
        for j in range(self.X_test.shape[1]):
          # Xác suất của đặc trưng i xảy ra khi
          P_featrurej_labeli = np.log(1/np.sqrt(2*np.pi*self.matrix_var[i,j])) -(self.X_test[0,j]-self.matrix_mean[i,j])**2/(2*self.matrix_var[i,j])
          self.P_labels_X_test[i]+= P_featrurej_labeli
      # Tìm giá trị log lớn nhất
      # Kỹ thuật Log sum exp trick -> tránh đưa exp(-1000) về 0
      c = self.P_labels_X_test.max()
      self.P_labels_X_test = np.exp(self.P_labels_X_test-c)
      self.P_labels_X_test_normalization = self.P_labels_X_test/self.P_labels_X_test.sum()
      return self.Labels_unique[np.argmax(self.P_labels_X_test_normalization)]


#-------------------------------------------------------------------------------
class K_means:
  def __init__(self,n_cluster = 8,max_iter = 100):
    self.n_cluster = n_cluster
    self.max_iter = max_iter
  def fit(self,X):
    if X.ndim ==1:
      raise ValueError("X must be 2D")
    else:
      self.X = X
      self.new_centroids = X[np.random.choice(np.arange(X.shape[0]),self.n_cluster)] # Tạo ra K tâm cụm khởi tạo ngẫu ngiên
      self.old_centroids = np.zeros_like(self.new_centroids)
      self.matrix_distance = np.stack(((np.ones((X.shape[0],1)),)*self.n_cluster),axis=2)
      iter = 0
      while not np.allclose(self.new_centroids,self.old_centroids) and iter < self.max_iter:
        self.old_centroids = np.copy(self.new_centroids)
        for i in range(self.n_cluster):
          self.matrix_distance[:,:,i] = np.sqrt(np.sum((self.X-self.new_centroids[i])**2,axis=1)).reshape(-1,1) # Tính ma trận khoảng cách từ 1 điểm đến tất cả các điểm
        self.idx_matrix = np.argmin(self.matrix_distance,axis=2).flatten()
        self.new_centroids = np.array([np.mean(self.X[self.idx_matrix == i],axis=0) for i in range(self.n_cluster)])
        iter += 1
      self.Labels = self.idx_matrix.reshape(-1,1)
      self.last_centroids = self.new_centroids
  def get_centroids(self):
    return self.last_centroids
  def get_label(self):
    return self.Labels
  def predict(self,X_test):
    if X_test.ndim ==1:
      raise ValueError("X must be 2D")
    else:
      self.X_test = X_test
      self.matrix_distance = np.stack(((np.ones((X_test.shape[0],1)),)*self.n_cluster),axis=2)
      for i in range(self.n_cluster):
        self.matrix_distance[:,:,i] = np.sqrt(np.sum((self.X_test-self.last_centroids[i])**2,axis=1)).reshape(-1,1)
      self.pred = np.argmin(self.matrix_distance,axis=2)
      return self.pred