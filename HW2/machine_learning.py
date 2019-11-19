"""
File to contain LWR Class
"""
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

        self.W = np.zeros([self.N,self.N])

        self.beta = 0
        self.residuals = 0
        self.h = h # bandwidth

        self.Zinv = 0

    def determine_beta(self, input_mat, output_vec, inv_matrix=None):

        if inv_matrix is None:
            buf = np.dot(input_mat.T, input_mat)
            buf = np.linalg.pinv(buf)
        else:
            buf = inv_matrix

        buf = np.dot(buf, input_mat.T)
        B = np.dot(buf, output_vec)

        return B

    def unweighted_prediction(self, query):

        self.beta = self.determine_beta(self.training_data_x, self.training_data_y)
        uwprediction = np.dot(query, self.beta)

        return uwprediction

    def weighted_prediction(self,query):
        """
        Calculate the weighted prediction for a given query array
        """

        self.get_weight_matrix(query)

        Z_matrix = np.dot(self.W, self.training_data_x)
        V_matrix = np.dot(self.W, self.training_data_y)

        buf = np.dot(Z_matrix.T, Z_matrix)
        buf = np.linalg.pinv(buf)

        self.Zinv = np.copy(buf)

        buf = np.dot(query.T, buf)
        buf = np.dot(buf,Z_matrix.T)
        wprediction = np.dot(buf,V_matrix)

        self.beta = self.determine_beta(Z_matrix, V_matrix, inv_matrix=self.Zinv)

        return wprediction

    def get_weight_matrix(self, query):
        """
        Determine the weight matrix for a query point
        """

        self.n_lwr = 0
        self.criteria = 0

        for i, xi in enumerate(self.training_data_x):
            buf = np.dot((xi - query).T,(xi - query)) # Euclidean Distance
            d = np.sqrt(buf) / self.h # Euclidean Distance
            K = np.exp(-1 * np.dot(d.T, d)) # Gaussian Kernal
            self.W[i][i] = round(np.sqrt(K), 4) # Assemble Weight Matrix

    def evaluate_learning(self):

        n_lwr = 0
        p_lwr = 0
        criteria = 0
        MSE_cv = 0

        for i, xi in enumerate(self.training_data_x):

            if self.W[i][i] == 0:
                continue

            else:
                z_i = self.W[i][i] * xi
                v_i = self.W[i][i] * self.training_data_y[i]

                r_i =  np.dot(z_i, self.beta) - v_i

                buf = self.W[i][i]**2
                buf = np.dot(buf, z_i.T)
                buf = np.dot(buf, self.Zinv)
                p_lwr = np.dot(buf, z_i)

                criteria += r_i**2
                n_lwr += self.W[i][i]**2

                buf = np.dot(z_i.T, self.Zinv)
                buf = np.dot(buf, z_i)

                MSE_cv += (r_i / (1 - buf))**2

        var_q = criteria / (n_lwr - p_lwr)
        MSE_cv_q = MSE_cv * (1 / n_lwr)

        return var_q, MSE_cv_q
