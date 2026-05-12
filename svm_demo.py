import numpy as np
from enum import Enum

# Kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))

MAX_OUTER_ITERS = 10000
MAX_PASSES = 20

class SVM_model:
    class KernelType (Enum):
        LINEAR_KERNEL = 1
        POLYNOMIAL_KERNEL = 2
        RBF_KERNEL = 3
    
    def __init__(self, input_batch, output_labels, kernel_type = KernelType.LINEAR_KERNEL, tol=1e-3, C=1.0, gamma=0.1, coef0=0.1, degree=3):
        self.K = []
        self.kernel_type = kernel_type
        
        self.X = np.asarray(input_batch)
        self.y = np.asarray(output_labels)
        self.b = 0.0

        if self.X.ndim != 2:
            raise ValueError("Input batch must be a 2D array of shape (n_samples, n_features).")

        if self.y.ndim != 1:
            raise ValueError("Output labels must be a 1D array of shape (n_samples,).")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X has {self.X.shape[0]} samples but y has {self.y.shape[0]} labels.")

        self.feature_length = self.X.shape[1]

        self.tol = tol
        self.C = C

        self.alpha = np.zeros(self.X.shape[0])
        self.support_vectors = []
        self.support_labels = []
        self.support_alphas = []

        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

        self.K = self.compute_kernel_matrix(self.X)


    # returns the decision function value for all input samples
    def decision_function(self):
        v = self.alpha * self.y             # (m,)
        return self.K @ v + self.b  # (m,)


    def verify_kkt_conditions(self, y, f_i, i):
        residual = f_i * self.y[i] - 1
        # We need to check if alpha is 0 < alpha_i < C. If this is true then the point lies on the margin. Then
        # self.y[i] * f[i] = 1 <==> residual = 0. Consider the following cases (which enforce 0 < alpha_i < C):

        # If alpha_i < C - tolerance (the alpha is below the upperbound) KKT then states that it is on the margin or outside (self.y[i] * f[i] >= 1).
        # If the point is within the margin then self.y[i] * f[i] < 1 then there is a violation
        if self.alpha[i] < self.C and residual < self.tol * -1 or self.alpha[i] > 0 and residual > self.tol: # VIOLATED CONDITIONS
            return False
        return True

    def compute_margin(self):
        passes = 0
        iterations = 0
        while passes < MAX_PASSES and iterations < MAX_OUTER_ITERS:
            '''indices = self.verify_kkt_conditions()

            if not indices:
                print("Entered")
                break'''
            
            num_alpha_changed = 0
            #f = self.decision_function()

            for i in range(self.X.shape[0]):

                #f_i = np.dot(alpha[i] * y, K[:, i]) + b
                #E_i = f_i - self.y[i]
                
                # f = self.decision_function() # MAKES IT 0(N^3) SO SLOW
                f_i = np.dot(self.alpha * self.y, self.K[:, i]) + self.b
                E_i = f_i - self.y[i]
                #E_i = f[i] - self.y[i]

                if self.verify_kkt_conditions(self.y[i], f_i, i):
                    continue

                
                '''E_i = f_i - self.y[i]
                #E_i = f_i - self.y[i]

                
                curr_delta = -np.inf
                best_j = None
                best_E_j = None

                for j in range(len(self.y)):
                    if j == i:
                        continue
                    f_j = np.dot(self.alpha * self.y, self.K[:, j]) + self.b
                    E_j = f_j - self.y[j]
                    delta = abs(E_j - E_i)
                    print("Delta=", delta)
                    if delta > curr_delta:
                        curr_delta = delta
                        best_j = j
                        best_E_j = E_j

                j = best_j
                E_j = best_E_j

                print(j, ", ", E_j)'''


                
                j = i
                while j == i:
                    j = np.random.randint(0, self.X.shape[0])
                f_j = np.dot(self.alpha * self.y, self.K[:, j]) + self.b
                E_j = f_j - self.y[j]



                # Compute lower and higher bound
                L = 0
                H = 0

                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]

                # Compute L and H
                if self.y[i] != self.y[j]:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)

                
                if L == H:
                    continue

                # Now that we select the pair (alpha_i, alpha_j), we compute the new multipliers
                # Compute the 2nd derivative (the curvature)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j
                self.alpha[j] = alpha_j_old - self.y[j] * (E_i - E_j) / eta

                # Clip to [L, H]
                if self.alpha[j] > H:
                    self.alpha[j] = H
                elif self.alpha[j] < L:
                    self.alpha[j] = L

                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    self.alpha[j] = alpha_j_old
                    continue

                # Update alpha_i
                self.alpha[i] = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                # Update threshold b
                b1 = (self.b - E_i- self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i]- self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j])

                b2 = (self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j])

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

                num_alpha_changed += 1

            if num_alpha_changed == 0:
                passes += 1
            else:
                passes = 0
            iterations += 1
            
        self.support_vectors = []
        self.support_alphas = []
        self.support_labels = []
        # support vectors are alpha_i>0 or in this case alpha_i > tolerance
        for i in range(len(self.y)):
            if self.alpha[i] > self.tol:
                self.support_vectors.append(self.X[i])
                self.support_alphas.append(self.alpha[i])
                self.support_labels.append(self.y[i])
        
    
    def compute_raw_data(self, X):
        v = self.alpha * self.y
        K = self.compute_kernel_matrix(X, self.X)
        return K @ v + self.b

    def predict(self, X):
        v = self.alpha * self.y
        K = self.compute_kernel_matrix(X, self.X)
        f = K @ v + self.b

        y = np.zeros(f.shape[0])

        for i in range(f.shape[0]):
            if f[i] > 0:
                y[i] = 1
            elif f[i] < 0:
                y[i] = -1
            else:
                y[i] = 0

        return y



    def compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                if self.kernel_type == self.KernelType.LINEAR_KERNEL:
                    K[i][j] = linear_kernel(X1[i], X2[j])

                elif self.kernel_type == self.KernelType.POLYNOMIAL_KERNEL:
                    K[i][j] = polynomial_kernel(
                        X1[i], X2[j],
                        degree=self.degree,
                        coef0=self.coef0)

                elif self.kernel_type == self.KernelType.RBF_KERNEL:
                    K[i][j] = rbf_kernel(X1[i], X2[j], gamma=self.gamma)

                else:
                    raise ValueError(f"Unknown kernel type: {self.kernel_type}")


        return K
       

        



