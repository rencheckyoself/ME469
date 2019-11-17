import numpy as np

# import data and assign each control and GT data point a unique ID
def load_data(file_name, header, footer, cols):
    """Loads in data from a text file
       Function to import data from a file

       Input:
            file_name = Full file names
            header = number of header rows starting from 0
            footer = number of footer rows starting from 0
            cols = select the columns with useful data. set to None for all columns

       Output:
            data = list type of data from file.
       Files must be in same directory as the python file."""

    data = np.genfromtxt(file_name, skip_header=header, skip_footer=footer,
                         names=True, dtype=None, delimiter=' ', usecols=cols)
    return data

class LWLR(object):
    """
    Class to handle LWLR learning for given data set.
    """
    def __init__(self, data_x, data_y, h):
        self.training_data_x = data_x
        self.training_data_y = data_y

        self.N = data_x.shape[0]
        self.n = data_x.shape[1]

        self.beta = self.determine_beta()

        # self.p = 2 # kernal function exp
        self.h = h # bandwidth

    def determine_beta(self):

        buf = np.dot(self.training_data_x.T, self.training_data_x)
        buf = np.linalg.inv(buf)
        buf = np.dot(buf,self.training_data_x.T)

        B = np.dot(buf, self.training_data_y)

        return B

    def unweighted_prediction(self, query):

        uwprediction = np.dot(query, self.beta)

        return uwprediction

    def weighted_prediction(self,query):
        """
        Calculate the weighted prediction for a given query array
        """
        W_matrix = self.get_weight_matrix(query)

        Z_matrix = np.dot(W_matrix, self.training_data_x)
        V_matrix = np.dot(W_matrix, self.training_data_y)

        buf = np.dot(Z_matrix.T, Z_matrix)
        buf = np.linalg.pinv(buf)
        buf = np.dot(query.T, buf)
        buf = np.dot(buf,Z_matrix.T)
        wprediction = np.dot(buf,V_matrix)

        return wprediction

    def get_weight_matrix(self, query):
        """
        Determine the weight matrix for a query point
        """
        W = np.zeros([self.N,self.N])

        for i, xi in enumerate(self.training_data_x):
            buf = np.dot((xi - query).T,(xi - query)) # Euclidean Distance
            d = np.sqrt(buf) / self.h # Euclidean Distance
            K = np.exp(-np.dot(d.T, d)) # Gaussian Kernal
            W[i][i] = round(np.sqrt(K), 4) # Assemble Weight Matrix

        return W
