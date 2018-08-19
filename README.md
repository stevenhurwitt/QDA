# QDA
Class to implement training and prediction using quadratic discriminant analysis.

Train method takes response y, data matrix X and number of classes of the response k.
It outputs a numpy array of the mean vector, covariance matrix and probability of each class for each k class.

Predict method predicts the class by calculating:

![alt text](https://github.com/stevenhurwitt/QDA/raw/master/images/QDA.png "")      

where the predicted class is the one that minimizes this over all k.

The main function shows how to implement this class.
