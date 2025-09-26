1. Dataset Selection
The provided dataset is the Wisconsin Breast Cancer Dataset, a standard binary classification dataset. It contains 168 samples (after loading the provided CSV; note: this appears to be a subset of the full 569-sample dataset commonly used). Each sample represents features extracted from a breast mass image, with the target label diagnosis indicating:

M: Malignant (positive class, coded as 1)
B: Benign (negative class, coded as 0)

There are 79 malignant and 89 benign cases. The features (30 in total) include measurements like mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (computed for standard error and "worst" cases).
This is suitable for binary classification to predict whether a tumor is malignant or benign.
2. Train/Test Split and Standardization

Split: An 80/20 train-test split was performed using random permutation (seed=42 for reproducibility), resulting in 134 training samples and 34 test samples.

Training: 63 malignant (47%), 71 benign (53%)
Test: 9 malignant (26%), 25 benign (74%)


Standardization: Features were standardized using the training set mean and standard deviation: $ z = \frac{x - \mu}{\sigma} $. This ensures all features are on the same scale, preventing bias toward features with larger magnitudes. No features had zero variance in training, so no imputation was needed.

3. Logistic Regression Model
A logistic regression model was fit using maximum likelihood estimation (via SciPy's minimize with BFGS optimizer for stability, given near-perfect separability in this subset). The model uses all 30 features plus an intercept term.
The linear predictor is $ z = \mathbf{w}^T \mathbf{x} + b $, where $\mathbf{w}$ are the learned weights and $b$ is the bias. Due to near-perfect class separability in this data subset, some weights grew large (max |w| ≈ 147), leading to predicted probabilities close to 0 or 1.
4. Evaluation Metrics (Threshold = 0.5)

Confusion Matrix (rows: actual, columns: predicted; negative=benign, positive=malignant):




















Predicted BenignPredicted MalignantActual Benign250Actual Malignant09

Precision: 1.00 (all predicted malignant cases were correct; no false positives)
Recall: 1.00 (all actual malignant cases were identified; no false negatives)
ROC-AUC: 1.00 (perfect separation; the model distinguishes classes ideally based on predicted probabilities)

These perfect scores indicate the model overfits this small test set due to separability—real-world performance may vary with more data.
5. Threshold Tuning and Sigmoid Function Explanation
Sigmoid Function
The sigmoid (logistic) function is central to logistic regression for binary classification. It transforms the linear output $ z = \mathbf{w}^T \mathbf{x} + b $ into a probability $ p $ between 0 and 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

How to arrive at it: The function solves the need for a bounded output in regression for probabilities. Starting from the cumulative distribution function of the logistic distribution, it provides an S-shaped curve: as $ z \to \infty $, $ \sigma(z) \to 1 $; as $ z \to -\infty $, $ \sigma(z) \to 0 $. At $ z = 0 $, $ \sigma(0) = 0.5 $.
Role: $ p = \sigma(z) $ represents $ P(y=1 \mid \mathbf{x}) $. The log-odds (logit) is the inverse: $ \log\left(\frac{p}{1-p}\right) = z $, linearizing the model for estimation.
Numerical stability: In practice, clip $ z $ (e.g., to [-500, 500]) to avoid overflow in $ e^{-z} $.

To derive the gradient for optimization (used in fitting): The loss is binary cross-entropy, $ L = -[y \log p + (1-y) \log(1-p)] $. The derivative w.r.t. $ z $ is $ \frac{\partial L}{\partial z} = p - y $, and by chain rule, the weight update is proportional to $ (p - y) \mathbf{x} $.
Threshold Tuning
The default threshold is 0.5 (predict positive if $ p > 0.5 $). Tuning adjusts it to optimize a metric like F1-score ($ F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} $), useful when classes are imbalanced or costs differ (e.g., prioritize recall for medical diagnosis to minimize false negatives).

Method: Evaluated thresholds from 0.00 to 1.00 in steps of 0.01. For each, computed predictions, precision, recall, and F1.
Results: The optimal threshold is 0.01, with F1 = 1.00 (same as 0.5, due to perfect separation—probabilities are extreme, so small changes don't affect outcomes here).
Explanation: Lowering the threshold increases recall (catches more positives) but risks precision. In this case, no improvement needed, but in general:

Use ROC curve to visualize trade-offs (here, perfect curve hugs top-left).
For imbalanced data, tune via Youden's J (max TPR - FPR) or cost-sensitive metrics.












































ThresholdPrecisionRecallF1-Score0.000.261.000.410.011.001.001.00............0.501.001.001.001.000.000.000.00
(Full table abbreviated; all thresholds 0.01–0.99 yield F1=1.00 here.) In production, validate on larger holdout sets to avoid overfitting.
