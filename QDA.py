import os
import math
import numpy as np

#Goal: Quadratic Discriminant Analysis
#to predict whether data is one of 11 vowels

##QDA: using training data,
#find mean vector & covariance matrix for each k class
#form discriminant function for each class
#d(x) = -.5log(determinant(cov_mat)) + (x-mu).T * inv(cov_mat) * (x-mu) + log(p_k)

#class to train QDA on X and y with k classes
class QDA:
    
    #save mean vectors, covariance matrix and probability for each class
    def train(self, y, X, k):
        self.y = y
        self.X = X
        self.k = k
        self.n = len(y)
        
        means = []
        cov = []
        prob = []

        for j in range(0,self.k):
            index = (self.y == (j+1))
            y_k = self.y[index]
            X_i = self.X[index]
    
            means.append(X_i.mean(axis=0))

            cov_mat = np.cov(X_i.T)
            cov.append(cov_mat) #try eigen decomp on cov mat
            prob.append(math.log(len(y_k)/self.n))

        self.means = np.array(means)
        self.cov = np.array(cov)
        self.prob = np.array(prob)
        return self.means, self.cov, self.prob

    #calculate discriminant function for all k classes
    def discrim(self, x, i):
        self.x = x
        self.i = i

        means, cov, prob = self.train(self.y, self.X, self.k)
        
        diff = (self.x - means[self.i])
        log_det = np.linalg.slogdet(cov[self.i])[1]
        std_mean_diff = (diff.T.dot(np.linalg.inv(cov[self.i]))).dot(diff)
        return (log_det + std_mean_diff - 2*prob[self.i])

    #return index of max class
    def pred(self, X_test):
        self.X_test = X_test
        self.m = self.X_test.shape[0]
        pred_classes = []

        for l in range(0,self.m):
            x_vec = X_test[l]
            discrim_vec = [0]*self.k
            
            for j in range(0,self.k):
                discrim_vec[j] = self.discrim(x_vec, j)

            pred_class = np.argmin(np.array(discrim_vec))+1
            pred_classes.append(pred_class)
                
        return np.array(pred_classes)

def main():
    os.chdir('/Volumes/YPNHS/Python Code/ML Classes')
    vowel_train = np.loadtxt('vowel_train.csv', delimiter=',', skiprows=1)
    vowel_test = np.loadtxt('vowel_test.csv', delimiter=',', skiprows=1)

    y_tr = vowel_train[:,0]
    X_tr = vowel_train[:,1:]

    y_test = vowel_test[:,0]
    X_test = vowel_test[:,1:]

    model = QDA()
    model.train(y_tr, X_tr, 11)
    print(model.pred(X_test))

    
if __name__ == '__main__':
    main()


