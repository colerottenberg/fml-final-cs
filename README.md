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
    2. Principal Component Analysis(PCA)
    3. PCA vs LDA (unsupervised vs supervised)
    4. Multidimensional Scaling (MDS)
    5. ISOMAP (Isometric Mapping)
    6. Locally Linear Embedding (LLE)
4. Artificial Neural Networks (Lecture 25 - 29)
    1. Multi-Layer Perceptron (MLP)
    2. Universal Approximation Theorem
    3. Activation functions: ReLU, leaky ReLu, sigmoid, tanh, softmax, linear, etc.
    4. Backpropagation
    5. Common challenges in training ANNs and strategies to address them
    6. Vanishing/exploding gradients
    7. Learning curves
    8. Network architecture
    9. Output Encoding: integer, one-hot, binary
    10. Optimization Techniques with Gradient Descent
        * Accelerated Gradient Descent strategies: momentum, Nesterrov's momentum
        * Adaptive Learning Rate: Adam
        * Learning rate schedulers
    11. Early stopping criteria
    12. Online vs Batch vs Mini-Batch learning
    13. Weight initializations strategies
    14. Stopping Criteria
    15. Data scaling/normalization
    16. Network pruning via regularization and dropout
    17. Batch normalization
    18. Determining whether to gather more data
5. Deep Learning (Lecture 28-29)
    1. Deep learning vs machine learning
    2. Convolutional layer vs dense layer
    3. Convolutional Neural Networks (CNNs)
    4. Pooling layers
    5. Strides
    6. Transfer learning

### Assignments to Review:
- HW3
- HW4
- SA4
- SA5

### Practice Problems from Exams:

## REVIEW

Naive Bayes:
    * Type: Generative
    * Mapper: i = $arg_{k} max P(C_{k} | \mathbf{x}) = arg_{k} max \frac{P(\mathbf{x} | C_{k})P(C_{k})}{P(\mathbf{X})}$
    * Objective Function: Find the label that maximizes the posterior probability. This requires learning the data likelihood.
    * Assumptions: Assume each class has a Gaussian distribution.
    * Complexity: If we have K classes and d dimensions, the complexity is O(Kd).
    * Sensitivity: Sensitive to Outliers
    * Convergence: Unique guarantee of convergence

Fisher's Linear Discriminant Analysis:
    * Type: Discriminative
    * Mapper: $$y(x) = \begin{cases} 1 & \text{if } w^{T}x + w_{0} > 0 \\ 0 & \text{otherwise} \end{cases} $$
    * Objective Function: $$ J(\mathbf{w}, w_{0}) = \frac{w^{T}S_{B}w}{w^{T}S_{W}w} $$, where $S_{B}$ is the between-class scatter matrix and $S_{W}$ is the within-class scatter matrix.
    * Learning Algorithm: **Eigendecomposition** of $S_{W}^{-1}S_{B}$. This updates the weights.
    * Assumptions: Assumes Gaussian distribution of data.
    * Complexity: If sample size, N, is much larger than the dimensionality, d, then the complexity is $O(d^2 N)$, otherwise it is $O(d^3)$.
    * Sensitivity: Sensitive to Outliers, hyperplane
    * Convergence: Unique guarantee of convergence assuming inverse exists.
### Perceptron:
    * Type: Discriminative
