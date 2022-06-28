### Interview Readiness
1. What is Normalization and how does Normalization make training a model more stable? <br>
    - Normalization is when we translate the date into some range (typically [0,1]) - also known as midmax scaling
    - we typically normalize our data so we can compare it on a common scale. For instance, some units may have massive units ,like e10lbs vs lbs, but may appear to be between 0 and 1. We want to make sure we're able to capture the features on a leveled scale where their true distributions can be captures. 
    
2. Loss and Optimizers 
    - A) What are loss and optimizer functions and how do they work?
        - loss and optimizer functions describe how the Machine or Deep learning models change their weights and reach a prediction.
        - For instance, an optimizer function is used to optimize the weights of the model such that we reduce loss while reducing over/underfitting and increase our accuracy. 
        - loss is the function which we're attempting to minimize during learning. The most common is Gradient Descent
        
    - B) What is Gradient Descent and how does it work?
        - Gradient Descent is an optimization function for finding the local minimum of a differentiable function. The idea is that if we've changed our weights in such a way that we've reduced our gradient descen, we're approaching a local minimum. We can continue to adjust our weights in such as way until we get a constant (or diminishing returns) and we've found our minimum.  
        
3. Activation Functions 
    - A) What is an activation function?
        - Also known as nonlinearities, activation functions are nonlinear equations which define the output of a node (or model) given a set of inputs and transforming them to some meaningful output
    - B) What are the outputs of the following activation functions: ReLU, Softmax Tanh, Sigmoid
        - ReLU: Rectified Linear Unit, this outputs the max between (0, z) where z is some set of inputs
            - $relu = max(0,z)$
        - Softmax: the softmax function, or normalized expontential function, is a generalization of a logistic function to multiple dimensions. In plain speak, it inputs a series of inputs and then returns a probability distribution of size inputs. 
            - $\sigma(z)_i = \frac{e^{z_i}}{\Sigma^{K}_{i}e^{\Betaz_j}} $
        - tanh: This activation maps between -1 and 1, with more positive values being 1 and small value (or more negative) go to -1. This results in strong gradient values during training and higher (more aggressive) updates in the weights of the network. 
            - $tanh(z) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
        - Sigmoid: similar to tanh, this maps large positive values to 1, but smaller (or negative) values approach 0 instead of -1. It has a less aggressive gradient, but can saturate and kill gradients if it goes unstable. I typically don't see this used much outside of binary classifiers in ML (not so much DL) 
            - $ S(x) = \frac{1}{1+e^{-x}}$
            
4.TPOT Algorith: 
    - A) What is the TPOT algorithm and how does it work?
        - TPOT is an AutoML tool specifically designed for the efficient construction of optimal pipelines through genetic programming. It works for supervised learning and automates everything from the feature selection/preprocessing/engineering all the way through Parameter optimization. 
        - It does this in a tree based way, trying different models, selection/engineering, ect til it finds a "tree" that provides an optimal max classification accuracy.
    - B) What does TPOT stand for?
        - TPOT stands for Tree-based Pipeline Optimization Tool