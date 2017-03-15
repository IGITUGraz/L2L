import numpy as np

GAUSSIAN_DISTRIBUTION = 'Gaussian'

class Distribution():
    def __init__(self, name):
        self.name = name
        pass
 
    def fit(self, data_list):
        pass
 
    def sample(self, n_items):
        pass
     
    def name(self):
        return self.name
    
class Gaussian(Distribution):
    
    def __init__(self, initial_data):
        super().__init__(GAUSSIAN_DISTRIBUTION)
        self.mean = np.zeros(initial_data[0].shape)
        self.cov = np.zeros((initial_data[0].shape[0], initial_data[0].shape[0]))
        self.fit(initial_data)
  
    def fit(self, data_list, smooth_update=1):
        """fit a normal distribution to the given data
        :param data_list: list or numpy array with individuals as rows
        """
        mean = np.mean(data_list, axis=0)
        # lookup np.cov
        cov_mat = np.cov(data_list, rowvar=False)
 
        self.mean = smooth_update * mean + (1 - smooth_update) * self.mean
        self.cov = smooth_update * cov_mat + (1 - smooth_update) * self.cov
 
    def sample(self, n_items):
        return np.random.multivariate_normal(self.mean, self.cov, n_items)
    
DISTRIBUTION_DICT = {GAUSSIAN_DISTRIBUTION : Gaussian}