import numpy as np
import time

#Error
def _mean_squared_error(y, pred):
    return 0.5 * np.mean((y - pred) ** 2)

def _mean_abs_error(y, pred):
    return np.mean(np.abs(y, pred))

#activate function
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _fourier(x):
    return np.sin(x)

def _identity(x):
    return x

def _hardlimit(x):
    return (x >= 0).astype(int)

#Get function
def getActivation(name):
    return {
        'sigmoid': _sigmoid,
        'fourier': _fourier,
        'hardlimit': _hardlimit,
    }[name]

def getLoss(name):
    return {
        'mse': _mean_squared_error,
        'mae': _mean_abs_error
    }[name]

#T function
def T(x,L1,L2,cn):
    r = x - np.dot(L1,x) + L2
    s = np.abs(r)-cn
    s = np.maximum(s,0,s)
    return s*np.sign(r)

def thetan(x0,x1,n):
    if (x0==x1).all():
        return 0
    else:
        return 1/(2**n*np.linalg.norm(x1-x0,'fro'))


class PISFPA:
    def __init__(self, num_input_nodes, num_hidden_units, num_output_units, activation='sigmoid',loss='mse', beta_init=None, w_init=None, bias_init=None):
        self._num_input_nodes = num_input_nodes
        self._num_hidden_units = num_hidden_units
        self._num_output_units = num_output_units

        self._activation = getActivation(activation)
        self._loss = getLoss(loss)

        #weight_out
        if isinstance(beta_init, np.ndarray):
            self._beta = beta_init
        else:
            self._beta = np.random.uniform(-1., 1., (self._num_hidden_units, self._num_output_units))

        #weight_in
        if isinstance(w_init, np.ndarray):
            self._w = w_init
        else:
            self._w = np.random.uniform(-1., 1., (self._num_input_nodes, self._num_hidden_units))

        #bias
        if isinstance(bias_init, np.ndarray):
            self._bias = bias_init
        else:
            self._bias = np.zeros(shape=(self._num_hidden_units,))

        print('Bias shape:', self._bias.shape)
        print('W shape:', self._w.shape)
        print('Beta shape', self._beta.shape)

    def fit(self, X, Y, itrs, lam, display_time=False):
        H = self._activation(X.dot(self._w) + self._bias)

        if display_time:
            start = time.time()

        L = 1. / np.max(np.linalg.eigvals(np.dot(H.T, H))).real
        m = H.shape[1]
        n = Y.shape[1]
        x0 = np.zeros((m,n))
        x1 = np.zeros((m,n))
        L1 = 2*L*np.dot(H.T, H)
        L2 = 2*L*np.dot(H.T, Y)

        for i in range(1,itrs+1):
            cn = ((2e-6*i)/(2*i+1))*lam*L
            beta = 0.9*i/(i+1)
            alpha = 0.9*i/(i+1)

            y = x1 + thetan(x0,x1,i)*(x1-x0)
            z = (1-beta)*x1 + beta*T(x1,L1,L2,cn)

            Ty = T(y,L1,L2,cn)
            Tz = T(z,L1,L2,cn)
            x = (1-alpha)*Ty + alpha*Tz

            x0, x1 = x1, x

        if display_time:
            stop = time.time()
            print(f'Train time: {stop-start}')

        self._beta = x

    def __call__(self, X):
        H = self._activation(X.dot(self._w) + self._bias)
        return H.dot(self._beta)

    def evaluate(self, X, Y):
        pred = self(X)

        loss = self._loss(Y, pred)

        acc = np.sum(np.argmax(pred, axis=-1) == np.argmax(Y, axis=-1)) / len(Y)

        return loss, acc
    
    

class ELM_AE:
    def __init__(self, n_hidden, X, activation='sigmoid',loss='mse'):
        self.X = X

        self._num_input_nodes = X.shape[1]
        self._num_output_units = X.shape[1]
        self._num_hidden_units = n_hidden
        
        self._activation = getActivation(activation)
        self._loss = getLoss(loss)
        
        self._beta = np.random.uniform(-1., 1., (self._num_hidden_units, self._num_output_units))
        self._w = np.random.uniform(-1., 1., (self._num_input_nodes, self._num_hidden_units))
        self._bias = np.zeros(shape=(self._num_hidden_units,))
        
        
    def fit(self, itrs, lam, display_time=False):
        H = self._activation(np.dot(self.X, self._w) + self._bias)

        if display_time:
            start = time.time()

        L = 1. / np.max(np.linalg.eigvals(np.dot(H.T, H))).real
        m = H.shape[1]
        n = self._num_output_units
        x0 = np.zeros((m,n))
        x1 = np.zeros((m,n))
        L1 = 2*L*np.dot(H.T, H)
        L2 = 2*L*np.dot(H.T, self.X)

        for i in range(1,itrs+1):
            cn = ((2e-6*i)/(2*i+1))*lam*L
            beta = 0.9*i/(i+1)
            alpha = 0.9*i/(i+1)

            y = x1 + thetan(x0,x1,i)*(x1-x0)
            z = (1-beta)*x1 + beta*T(x1,L1,L2,cn)

            Ty = T(y,L1,L2,cn)
            Tz = T(z,L1,L2,cn)
            x = (1-alpha)*Ty + alpha*Tz

            x0, x1 = x1, x

        if display_time:
            stop = time.time()
            print(f'Train time: {stop-start}')

        self._beta = x
        
    def __call__(self):
        H = self._activation( np.dot(self.X, self._w) + self._bias )
        return np.dot(H, self._beta)
