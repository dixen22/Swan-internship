# Internship Report
## Session 3 - Linear Regression and Gradient Descent
My final parameters are a learning rate of 0.0000255 for `a` and 0.82 for `b`. After 30 epochs, I obtain a minimum loss of 558.6971. This corresponds to weights of approximately 0.5553 for `a` and 94.5792 for `b`. The learning rates for `a` and `b` are not identical because the data has not been normalized. The calculation of the new `a` in gradient descent depends on the data; so, with big data, `a` changes much faster than `b`.

![](./Session3/img/LRPredictedLine.png) 
*Fig. 1 - The data and the line obtained from the regression model.*

As can be seen in *Fig. 1*, the data are not very linear and fare from the predicted line. This explains why the loss remains very high.

![](./Session3/img/LRLearningCurve.png) 
*Fig. 2 - The learning curve of model training.*

As can be seen in Fig. 2, the learning curve confirms that the model is training effectively. The loss, which was initially very high, decreases gradually and begins to converge around the 10th epoch.
