# Exam 1 Review

## Coverage
Coverage over Lectures 15 through 29

**Bring Calculator**

**8-9 Questions**

1. Discriminative Functions for Classifiers (Lecture 15 - 19)
    1. Performance Metrics: **ROC Curves, Confusion Matrices, F1 Score**
    2. Discriminative vs Probabilistic Classifiers
    3. Fisher's Linear Discriminant Analysis
    4. Logistic Regression
    5. Perceptron Algorithm
    6. Gradient Descent
2. Kernel Machine (Lecture 19 - 21)
    1. Kernel Machines
    2. RBF Kernel: **Infinite Dimension Feature Space**
    3. Kernel Trick
    4. Lagrange Optimization
    5. Hard Margin SVM
    6. Slack Variables
    7. Soft Margin SVM
3. Dimensionality Reduction (Lecture 22 - 25)
    1. Curse of Dimensionality
        - The curse of dimensionality refers to the phenomenon where the number of features in a dataset increases, the volume of the feature space increases exponentially. This leads to sparsity in the data and can lead to overfitting.
    2. Principal Component Analysis(PCA)
        - To find PCA of a dataset, we first center the data by subtracting the mean. Then, we find the covariance matrix of the centered data. We then find the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the principal components and the eigenvalues are the variance along the principal components. We can then project the data onto the principal components to reduce the dimensionality.
    3. PCA vs LDA (unsupervised vs supervised)
        - PCA is an unsupervised method that finds the principal components of the data. LDA is a supervised method that finds the linear discriminants that maximize the separation between classes.
    4. Multidimensional Scaling (MDS)
        - MDS is a technique that finds a low-dimensional representation of the data that preserves the pairwise distances between the data points. We find MDS using eigendecomposition of the distance matrix.
    5. ISOMAP (Isometric Mapping)
        - ISOMAP is similar to MDS but it uses the geodesic distance between points instead of the Euclidean distance. It uses the shortest path between points on a graph to find the distance between points.
    6. Locally Linear Embedding (LLE)
        - LLE is a technique that finds a low-dimensional representation of the data that preserves the local relationships between the data points. It does this by finding the weights that best reconstruct each point from its neighbors.
4. Artificial Neural Networks (Lecture 25 - 29)
    1. Multi-Layer Perceptron (MLP)
    2. Universal Approximation Theorem
    3. Activation functions: ReLU, leaky ReLu, sigmoid, tanh, softmax, linear, etc.
        - **ReLU**: $f(x) = max(0, x)$
        - **Leaky ReLU**: $f(x) = max(0.01x, x)$
        - **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$
        - **Tanh**: $f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
        - **Softmax**: $f(x) = \frac{e^{x_{i}}}{\sum_{j} e^{x_{j}}}$
        - **Linear**: $f(x) = x$
    4. Backpropagation
        - Backpropagation is a technique used to train neural networks. It involves computing the gradient of the loss function with respect to the weights of the network using the chain rule. The gradients are then used to update the weights using an optimization algorithm such as gradient descent.
    5. Common challenges in training ANNs and strategies to address them
    6. Vanishing/exploding gradients
        - With deep neural networks, the gradienet can become very small or very large as it is backpropagated through the network. For activation functions like sigmoid and tanh, the gradient can vanish as it is backpropagated through the network. For activation functions like ReLU, the gradient can explode as it is backpropagated through the network. This can make training the network difficult. Strategies to address this include using different activation functions, using batch normalization, and using techniques like residual connections.
    7. Learning curves
    8. Network architecture
    9. Output Encoding: integer, one-hot, binary
        - **Integer Encoding**: Assigns a unique integer to each class label.
        - **One-Hot Encoding**: Represents each class label as a binary vector where only one element is 1 and the rest are 0.
        - **Binary Encoding**: Represents each class label as a binary vector where each element is either 0 or 1.
    10. Optimization Techniques with Gradient Descent
        * Accelerated Gradient Descent strategies: momentum, Nesterrov's momentum
        * Adaptive Learning Rate: Adam
        * Learning rate schedulers
    11. Early stopping criteria
        - We can enable certain callbacks in the training process to stop the training early if certain conditions are met. For example, we can stop training if the validation loss does not improve for a certain number of epochs.
    12. Online vs Batch vs Mini-Batch learning
        - **Online Learning**: Update the weights after each training example. This is more sporadic and can be noisy.
        - **Batch Learning**: Update the weights after all training examples have been seen. This can be slow and requires a lot of memory.
        - **Mini-Batch Learning**: Update the weights after a subset of the training examples have been seen. This is a compromise between online and batch learning.
    13. Weight initializations strategies
        - **Random Initialization**: Initialize the weights randomly from a distribution such as a normal distribution or a uniform distribution.
        - **Xavier Initialization**: Initialize the weights using a normal distribution with mean 0 and variance $\frac{1}{n_{in}}$, where $n_{in}$ is the number of input units to the neuron.
    14. Stopping Criteria
    15. Data scaling/normalization
    16. Network pruning via regularization and dropout
        - **Dropout**: Dropout is a regularization technique where we randomly set a fraction of the neurons in the network to zero during training. This helps prevent overfitting by forcing the network to learn redundant representations. This leads to lower accuracy performance on training than validation.
    17. Batch normalization
        - **Batch Normalization**: Batch normalization is a technique used to normalize the inputs to a layer in a neural network. This helps stabilize the training process and can lead to faster convergence. It can also act as a form of regularization.
    18. Determining whether to gather more data
5. Deep Learning (Lecture 28-29)
    1. Deep learning vs machine learning
        - **Deep Learning** is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns in data. It is particularly well-suited for tasks such as image recognition, speech recognition, and natural language processing.
    2. Convolutional layer vs dense layer
        - **Convolutional Layer**: A convolutional layer applies a filter to the input data to extract features. It is commonly used in image processing tasks. This filter is also a learnable parameter.
    3. Convolutional Neural Networks (CNNs)
    4. Pooling layers
        - **Pooling Layer**: A pooling layer reduces the spatial dimensions of the input data by downsampling. This helps reduce the computational complexity of the network and makes it more robust to variations in the input.
    5. Strides
        - **Strides**: Strides refer to the number of pixels the filter moves across the input data. A stride of 1 means the filter moves one pixel at a time, while a stride of 2 means the filter moves two pixels at a time.
    6. Transfer learning
        - **Transfer Learning**: Transfer learning is a technique where we use a pre-trained model on a similar task as a starting point for training a new model. This can help speed up training and improve performance, especially when we have limited data. Transfer Learning can also be useful when we have a small dataset and we want to leverage the knowledge learned from a larger dataset.

### Assignments to Review:
- HW3
- HW4
- SA4
- SA5

### Practice Problems from Exams:

## REVIEW

Naive Bayes:
  - Type: Generative
  - Mapper: i = $arg_{k} max P(C_{k} | \mathbf{x}) = arg_{k} max \frac{P(\mathbf{x} | C_{k})P(C_{k})}{P(\mathbf{X})}$
  - Objective Function: Find the label that maximizes the posterior probability. This requires learning the data likelihood.
  - Assumptions: Assume each class has a Gaussian distribution.
  - Complexity: If we have K classes and d dimensions, the complexity is O(Kd).
  - Sensitivity: Sensitive to Outliers
  - Convergence: Unique guarantee of convergence

Fisher's Linear Discriminant Analysis:
  - Type: Discriminative
  - Mapper: $$y(x) = \begin{cases} 1 & \text{if } w^{T}x + w_{0} > 0 \\ 0 & \text{otherwise} \end{cases} $$
  - Objective Function: $$ J(\mathbf{w}, w_{0}) = \frac{w^{T}S_{B}w}{w^{T}S_{W}w} $$, where $S_{B}$ is the between-class scatter matrix and $S_{W}$ is the within-class scatter matrix.
  - Learning Algorithm: **Eigendecomposition** of $S_{W}^{-1}S_{B}$. This updates the weights.
  - Assumptions: Assumes Gaussian distribution of data.
  - Complexity: If sample size, N, is much larger than the dimensionality, d, then the complexity is $O(d^2 N)$, otherwise it is $O(d^3)$.
  - Sensitivity: Sensitive to Outliers, hyperplane
  - Convergence: Unique guarantee of convergence assuming inverse exists.

Perceptron:
  - Type: Discriminative
  - Mapper: $$y(x) = \phi(w^{T}x + w_{0})$$, where $\phi(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$
  - Objective Function: $$ \mathbf{E}_{p}(\mathbf{w}, w_{0}) = -\sum_{n \in \mathbf{M}} t_{n}(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0})$$, where $\mathbf{M}$ is the set of misclassified points.
  - Learning Algorithm: Stochastic Gradient Descent or Batch or Mini-Batch Gradient Descent
  - Assumptions: Assumes linearly separable data.
  - Complexity: $O(\frac{1}{\epsilon})$ where $\epsilon$ is the generalization error or the distance of the closest point to the decision boundary.
  - Sensitivity: Sensitive to Outliers
  - Convergence: Converges if data is linearly separable.

Logistic Regression:
  - Type: Probabilistic Discriminative
  - Mapper: $$y(x) = \begin{cases} 1 & \text{if } \phi(w^{T}x + w_{0}) > 0.5 \\ 0 & \text{otherwise} \end{cases} $$, where $\phi(x) = \frac{1}{1 + e^{-x}}$
  - Objective Function: $$J(\mathbf{w}, w_{0}) = -\sum_{n=1}^{N} t_{n} \log \phi(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}) + (1 - t_{n}) \log (1 - \phi(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}))$$, where $t_{n}$ is the target value.
  - Learning Algorithm: Gradient Descent
  - Assumptions: Assumes linearly separable data.
  - Complexity: $O(d)$
  - Sensitivity: Sensitive to Outliers
  - Convergence: Guaranteed global minimum provided a stable learning rate $\eta$ given the cross-entropy loss function is convex.

Hard-Margin SVM:
  - Type: Discriminative
  - Mapper: $$y(x) = \mathbf{w}^{T}\mathbf{x} + w_{0}$$
  - Objective Function: $$\mathcal{L}(\mathbf{w}, w_{0}, \mathbf{a}) = \frac{1}{2}||\mathbf{w}||^{2} - \sum_{n=1}^{N} a_{n}(t_{n}(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}) - 1), \mathbf{a}_n \geq \forall n$$, where $\mathbf{a}$ is the vector of Lagrange multipliers.
  - Learning Algorithm: Quadratic Programming
  - Assumptions: Assumes linearly seperable in feature mapping.
  - Complexity: $O(N^{3})$
  - Sensitivity: Sensitive to Outliers near the margin
  - Convergence: Guaranteed global minimum

Soft-Margin SVM:
  - Type: Discriminative
  - Mapper: $$y(x) = \mathbf{w}^{T}\mathbf{x} + w_{0}$$
  - Objective Function: $$\mathcal{L}(\mathbf{w}, w_{0}, \mathbf{a}, \mathbf{\mu}) = \frac{1}{2}||\mathbf{w}||^{2} + C\sum_{n=1}^{N} \xi_{n} - \sum_{n=1}^{N} a_{n}(t_{n}(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}) - 1) - \sum_{n=1}^{N} \mu_n \xi_n$$
  - Learning Algorithm: Quadratic Programming
  - Assumptions: Assumes linearly seperable in feature mapping.
  - Complexity: $O(N^{3})$
  - Sensitivity: Sensitive to Outliers near the margin
  - Convergence: Guaranteed global minimum
  - **Notes about Soft-Margin SVM**: The slack variables, $\xi_{n}$, are introduced to allow for misclassification. The hyperparameter, $C$, is used to control the trade-off between the margin and the misclassification. If C is larger, the model will began to recover the Hard-Margin SVM. If C is smaller, the model will allow for more misclassification.

### NAND Gate Problem
The NAND Gate problem revolves around a 2D plane with two features, $x_{0}$ and $x_{1}$. The NAND gate is a binary classification problem where the output is 1 if the input is not 1, 1 and 1, 0 and 1, and 1 and 0. The output is 0 if the input is 0 and 0. The decision boundary is a hyperplane that separates the two classes. The hyperplane is defined by the equation $w_{0} + w_{1}x_{1} + w_{2}x_{2} = 0$. The weights are learned through the perceptron algorithm. The perceptron algorithm updates the weights based on the misclassified points. The weights are updated by $w_{i} = w_{i} + \eta(t - y)x_{i}$, where $\eta$ is the learning rate, $t$ is the target value, $y$ is the predicted value, and $x_{i}$ is the feature value.

## Formula Sheet Rough Draft

### Discriminative Functions for Classifiers
Useful formulas for discriminative functions for classifiers.
  - Performance Metrics: **ROC Curves, Confusion Matrices, F1 Score**
      - **ROC Curves**: $TPR = \frac{TP}{TP + FN}$, $FPR = \frac{FP}{FP + TN}$
      - **Confusion Matrices**: $TP$, $FP$, $TN$, $FN$
      - **F1 Score**: $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
  - Fisher's Linear Discriminant Analysis
      - **Objective Function**: $$ J(\mathbf{w}, w_{0}) = \frac{w^{T}S_{B}w}{w^{T}S_{W}w} $$
      - **Learning Algorithm**: **Eigendecomposition** of $S_{W}^{-1}S_{B}$
      - The between-class scatter matrix, $S_{B}$, is defined as $S_{B} = \sum_{k=1}^{K} N_{k}(\mathbf{m}_{k} - \mathbf{m})(\mathbf{m}_{k} - \mathbf{m})^{T}$, where $N_{k}$ is the number of samples in class $k$, $\mathbf{m}_{k}$ is the mean of class $k$, and $\mathbf{m}$ is the overall mean.
      - To find the weights, we solve the generalized eigenvalue problem $S_{W}^{-1}S_{B}w = \lambda w$.
  - Logistic Regression
      - **Objective Function**: $$J(\mathbf{w}, w_{0}) = -\sum_{n=1}^{N} t_{n} \log \phi(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}) + (1 - t_{n}) \log (1 - \phi(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0}))$$
      - **Learning Algorithm**: Gradient Descent
      - **Activation Function**: $\phi(x) = \frac{1}{1 + e^{-x}}$
      - To find the weights, we solve the gradient of the objective function with respect to the weights.
  - Perceptron Algorithm
      - **Objective Function**: $$ \mathbf{E}_{p}(\mathbf{w}, w_{0}) = -\sum_{n \in \mathbf{M}} t_{n}(\mathbf{w}^{T}\mathbf{x}_{n} + w_{0})$$
      - **Learning Algorithm**: Stochastic Gradient Descent or Batch or Mini-Batch Gradient Descent
      - **Activation Function**: $\phi(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$
      - To find the weights, we update the weights based on the misclassified points.
