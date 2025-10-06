"""
Sections included:
- Foundational Neural Network Concepts
- Learning & Optimization Algorithms
- NLP & Transformers
- Related Concepts (weights, biases, learning rate, etc.)
- How modern LLM systems (ChatGPT, Grok) are structured
- Safe City & Drone system architectures
- Custom Full-Stack AI Developer Curriculum (detailed roadmap)
- Final notes and suggested next steps
- Expanded: Supervised, Unsupervised, and Reinforcement Learning
- Expanded: Classical ML Algorithms (KNN, Decision Trees, SVM, etc.)
- Expanded: Ensemble Methods
- Expanded: Clustering and Dimensionality Reduction
- Expanded: Convolutional Neural Networks (CNNs)
- Expanded: Recurrent Neural Networks (RNNs) and LSTMs
- Expanded: Generative Models (GANs, VAEs, Diffusion)
- Expanded: Advanced Topics (Transfer Learning, Federated Learning, Explainable AI, Quantum ML, etc.)
- New: Modern LLM Techniques and Prompt Engineering
- New: Graph Neural Networks and Specialized Architectures
- New: Meta-Learning and Advanced Learning Paradigms


"""

# -----------------------------
# PART 1: FOUNDATIONAL NEURAL NETWORK CONCEPTS
# -----------------------------

# Neural Network (Basic Structure)
# - Definition:
#   A neural network is a system inspired by the human brain, made up of layers of nodes (neurons)
#   that learn to recognize patterns in data. Each neuron receives inputs, processes them, and
#   passes output to the next layer.
#
# - Small working principle:
#   Each neuron multiplies inputs by weights, adds a bias, applies an activation function, and
#   passes the result forward. Layers of such neurons together form a network that can learn
#   complex relationships.
#
# - Flow of work:
#   1. Input data enters the input layer.
#   2. Each hidden layer transforms data using weights, biases, and activations.
#   3. Output layer produces predictions.
#   4. The process is repeated with updated weights during training.

# Weights and Biases
# - Definition:
#   Weights are adjustable parameters that determine the importance of each input. Biases allow
#   shifting the activation output, helping the model fit data better.
# - Small working principle:
#   Weights scale input values; biases shift them before applying the activation. Together, they
#   control how strongly a neuron reacts to a given input.
# - Flow of work:
#   1. Multiply input × weight.
#   2. Add bias.
#   3. Apply activation function → produces neuron output.
#   4. Adjust weights and biases during training to minimize error.

# Forward Propagation
# - Basic definition:
#   Forward propagation is the process of moving input data through the network to produce an
#   output (prediction).
# - Small working principle:
#   Each layer processes inputs using weights and biases, passes the result through an activation
#   function, and sends it forward to the next layer.
# - Flow of work:
#   1. Input data enters the first layer.
#   2. Each neuron computes (input × weight + bias).
#   3. Apply activation function → send result forward.
#   4. Final layer gives the model’s prediction.

# Activation Function (Sigmoid example)
# - Basic definition:
#   An activation function introduces non-linearity into the network, enabling it to learn
#   complex patterns. The Sigmoid function converts any real value into a range between 0 and 1.
# - Small working principle:
#   Sigmoid output = 1 / (1 + e^-x). If x is large positive → output close to 1; if x is large
#   negative → output close to 0.
# - Flow of work:
#   1. Compute neuron input (weighted sum).
#   2. Apply Sigmoid function → squash result between 0–1.
#   3. Pass this value to next layer or output.

# Loss Function / Cost Function
# - Basic definition:
#   A loss function measures how far the model’s prediction is from the actual target. A cost
#   function is the average loss over all training samples.
# - Small working principle:
#   If predictions are wrong, the loss is high; if correct, loss is low. The goal of training is
#   to minimize this loss.
# - Flow of work:
#   1. Model predicts output (via forward propagation).
#   2. Compare predicted output with true label.
#   3. Calculate loss (e.g., Mean Squared Error or Cross-Entropy).
#   4. Use this loss in backward propagation to adjust parameters.

# Backward Propagation (Backpropagation)
# - Basic definition:
#   Backpropagation is the process of calculating how much each weight contributed to the error and
#   updating them accordingly.
# - Small working principle:
#   It applies the chain rule from calculus to propagate the error backward through the network,
#   computing gradients (slopes) of the loss with respect to weights.
# - Flow of work:
#   1. Compute loss after forward propagation.
#   2. Calculate gradient of loss w.r.t output layer weights.
#   3. Move backward layer by layer, adjusting gradients.
#   4. Update weights to reduce future loss.

# -----------------------------
# PART 2: LEARNING & OPTIMIZATION ALGORITHMS
# -----------------------------

# Gradient Descent (GD)
# - Basic definition:
#   Gradient Descent is an optimization algorithm that updates weights in the opposite direction
#   of the loss gradient to minimize error.
# - Small working principle:
#   It computes the slope (gradient) of the loss with respect to weights and adjusts weights by a
#   small step (learning rate) toward the minimum loss.
# - Flow of work:
#   1. Calculate gradient (how fast loss changes with weights).
#   2. Update weights: new_weight = old_weight - learning_rate × gradient.
#   3. Repeat until loss is minimized.

# Stochastic Gradient Descent (SGD)
# - Basic definition:
#   SGD is a faster version of Gradient Descent that updates weights after every single training
#   sample instead of the entire dataset.
# - Small working principle:
#   Rather than using all data at once (batch), it uses random samples (stochastic) — making it
#   faster and adding noise that helps escape local minima.
# - Flow of work:
#   1. Pick one (or a few) random training sample(s).
#   2. Perform forward and backward propagation.
#   3. Update weights immediately.
#   4. Repeat for all samples (one epoch).

# Optimization (General field)
# - Definition:
#   Optimization is the process of finding the best set of parameters (weights and biases) that
#   minimize the loss function.
# - Small working principle:
#   It involves applying algorithms like Gradient Descent, Adam, or RMSprop to iteratively improve
#   performance by reducing error.
# - Flow of work:
#   1. Start with random weights.
#   2. Compute loss.
#   3. Apply optimization algorithm to update weights.
#   4. Continue until model performs well.

# Adam Optimizer
# - Basic definition:
#   Adam (Adaptive Moment Estimation) combines ideas from Momentum and RMSprop — adapting the
#   learning rate for each parameter and smoothing gradient updates.
# - Small working principle:
#   It keeps track of both the average of past gradients (momentum) and the average of squared
#   gradients (adaptive learning rate).
# - Flow of work:
#   1. Compute gradient.
#   2. Update moving averages of gradient (first moment) and squared gradient (second moment).
#   3. Use these to adaptively adjust learning rate for each parameter.
#   4. Update weights.

# AdamW Optimizer
# - Basic definition:
#   AdamW is an improved version of Adam that adds weight decay to reduce overfitting by penalizing
#   large weights.
# - Small working principle:
#   It separates weight decay (regularization) from gradient updates — providing better
#   generalization and stability.
# - Flow of work:
#   1. Perform Adam update steps.
#   2. Apply weight decay (reduce large weights slightly).
#   3. Continue iterative updates to minimize loss and prevent overfitting.

# RMSprop Optimizer
# - Basic definition:
#   RMSprop (Root Mean Square Propagation) is an optimizer that adapts the learning rate by dividing
#   the gradient by the root mean square of recent gradients, preventing exploding or vanishing gradients.
# - Small working principle:
#   It maintains a moving average of squared gradients to normalize the gradient updates, making it
#   effective for non-stationary objectives like in RNNs.
# - Flow of work:
#   1. Compute gradient.
#   2. Update the exponentially decaying average of squared gradients.
#   3. Adjust the learning rate by dividing the global learning rate by the square root of this average.
#   4. Update weights using the adjusted gradient.

# Momentum Optimizer
# - Basic definition:
#   Momentum is an extension of SGD that accelerates gradient descent by adding a fraction of the
#   previous update to the current update, helping to navigate ravines and avoid local minima.
# - Small working principle:
#   It accumulates velocity in directions of consistent gradients, smoothing out oscillations and
#   speeding up convergence.
# - Flow of work:
#   1. Compute current gradient.
#   2. Update velocity: velocity = beta * previous_velocity + (1 - beta) * gradient.
#   3. Update weights: weights = weights - learning_rate * velocity.
#   4. Repeat to build momentum in promising directions.

# -----------------------------
# PART 3: NATURAL LANGUAGE PROCESSING (NLP) & TRANSFORMERS
# -----------------------------

# Natural Language Processing (NLP)
# - Basic definition:
#   NLP is the field of AI that enables machines to understand, interpret, and generate human
#   language.
# - Small working principle:
#   It uses models to convert words into numerical forms, learn their relationships, and perform
#   tasks like translation, summarization, or question answering.
# - Flow of work:
#   1. Input text is tokenized (split into words/tokens).
#   2. Convert tokens to embeddings (numbers).
#   3. Feed into neural network (like RNN or Transformer).
#   4. Model outputs predictions (next word, sentiment, etc.).

# Embedding (Word Embeddings)
# - Basic definition:
#   Embeddings are numerical vector representations of words or tokens that capture their meanings
#   and relationships in a continuous space.
# - Small working principle:
#   Similar words have similar embeddings — e.g., “king” and “queen” have vectors close to each
#   other.
# - Flow of work:
#   1. Convert text → tokens.
#   2. Map each token → embedding vector.
#   3. Feed these vectors into the neural model for learning patterns.

# Transformer Architecture
# - Basic definition:
#   A Transformer is a deep learning model that processes sequences (like sentences) using
#   self-attention mechanisms instead of recurrence (RNNs).
# - Small working principle:
#   It looks at all words in a sentence simultaneously, learning which words relate to each other
#   more strongly.
# - Flow of work:
#   1. Input words → embeddings + positional encoding.
#   2. Apply self-attention to capture context.
#   3. Pass through feed-forward layers.
#   4. Stack multiple layers → final output (translation, etc.).

# Attention Mechanism (Self-Attention)
# - Basic definition:
#   Self-Attention allows a model to focus on relevant words when processing a given word,
#   improving understanding of context.
# - Small working principle:
#   It computes relationships between all pairs of words using queries, keys, and values —
#   assigning higher weights to more relevant words.
# - Flow of work:
#   1. Compute query, key, and value vectors for each word.
#   2. Compare query with all keys → get attention scores.
#   3. Multiply scores with values → weighted sum (context vector).
#   4. Pass context to next layer.

# Multi-Head Attention
# - Basic definition:
#   Multi-Head Attention runs multiple self-attention mechanisms in parallel, allowing the model
#   to capture different types of relationships simultaneously.
# - Small working principle:
#   Each "head" focuses on different aspects (e.g., syntax vs. semantics), and their outputs are
#   concatenated and linearly transformed.
# - Flow of work:
#   1. Split input into multiple heads.
#   2. Apply self-attention independently to each head.
#   3. Concatenate outputs from all heads.
#   4. Apply a linear layer to combine them.

# Positional Encoding
# - Basic definition:
#   Positional Encoding adds information about the position of tokens in a sequence to embeddings,
#   since Transformers lack inherent sequence order.
# - Small working principle:
#   It uses sine and cosine functions of different frequencies to encode positions, allowing the
#   model to learn relative positions.
# - Flow of work:
#   1. Generate positional vectors using sin/cos formulas for each dimension.
#   2. Add these vectors to the token embeddings.
#   3. Feed the combined embeddings into the Transformer layers.
#   4. The model learns to use this for sequence-aware processing.

# -----------------------------
# PART 4: ADDITIONAL RELEVANT CONCEPTS (FOR CONTEXT)
# -----------------------------

# Neural Network (revisited)
# - Basic structure with input, hidden, and output layers connected via weights and biases.
# - Multi-layer networks can learn hierarchical features from raw data.

# Learning Rate
# - Basic definition:
#   Controls how big a step is taken when updating weights during optimization.
# - Small working principle:
#   A high learning rate learns quickly but may overshoot the minimum; a low one learns slowly
#   but more precisely.
# - Flow of work:
#   1. Compute gradient.
#   2. Multiply gradient by learning rate.
#   3. Subtract result from weights.
#   4. Repeat every iteration.

# Epoch vs. Batch Size
# - Basic definition:
#   An epoch is one complete pass through the entire dataset.
#   Batch size is the number of samples processed before updating weights once.
# - Small working principle:
#   Large batches → stable but slow updates. Small batches → noisy but faster learning.
# - Flow of work:
#   1. Split data into batches.
#   2. Process each batch → update weights.
#   3. Repeat until all data seen once (1 epoch).
#   4. Continue for multiple epochs to improve accuracy.

# Overfitting & Underfitting
# - Basic definition:
#   Overfitting: Model memorizes training data but fails on new data.
#   Underfitting: Model is too simple to capture patterns in training data.
# - Small working principle:
#   Overfitting occurs when training loss is low but validation loss is high; underfitting when both losses are high.
# - Flow of work:
#   1. Train model → observe training and validation performance.
#   2. If overfitting → apply regularization, dropout, or more data.
#   3. If underfitting → use a larger model or train longer.

# Regularization (L1/L2)
# - Basic definition:
#   Regularization adds a penalty to the loss function to prevent overfitting by discouraging complex models.
#   L1 (Lasso) promotes sparsity; L2 (Ridge) shrinks weights evenly.
# - Small working principle:
#   It adds the sum (L1) or sum of squares (L2) of weights to the loss, forcing the optimizer to keep weights small.
# - Flow of work:
#   1. Compute standard loss.
#   2. Add regularization term: lambda * (sum|weights| for L1 or sum(weights^2) for L2).
#   3. Optimize the combined loss.
#   4. Results in simpler, more generalizable models.

# Dropout
# - Basic definition:
#   Dropout randomly ignores (drops out) a fraction of neurons during training to prevent co-dependency and overfitting.
# - Small working principle:
#   By dropping neurons, the model learns redundant representations, acting as an ensemble of sub-networks.
# - Flow of work:
#   1. During training, randomly set some neuron outputs to zero (with probability p).
#   2. Forward and backward propagate with the remaining neurons.
#   3. At inference, use all neurons but scale outputs by (1-p).
#   4. Improves robustness and generalization.

# -----------------------------
# PART 5: HOW MODERN LLM SYSTEMS WORK (ChatGPT, Grok, etc.)
# -----------------------------

# Core Architecture & Principles
# 1. Transformer Architecture: Core building block using self-attention + feed-forward layers.
# 2. Pre-training + Fine-tuning:
#    - Pre-training: Train on large corpora using self-supervised objectives (e.g., next-token prediction).
#    - Fine-tuning: Perform task-specific adjustments with supervised data and human feedback.
# 3. RLHF (Reinforcement Learning from Human Feedback):
#    - Humans rate outputs; a reward model is trained; the base model is fine-tuned with RL to maximize the reward.
# 4. Scalability & Deployment:
#    - Large models require distributed training/inference across many GPUs/TPUs.
#    - Techniques: quantization, pruning, caching, dynamic batching for lower latency.
# 5. Safety & Alignment:
#    - Filtering training data, safety layers, system prompts, content moderation, guardrails.
# 6. Tokenization, Embeddings, Positional Encoding:
#    - Convert text to tokens → map tokens to vectors → add positional info.
# 7. Inference/Decoding Strategies:
#    - Greedy, beam search, temperature sampling, top-k/top-p for text generation.

# Step-by-step Flow (Simplified)
# 1. Data Collection & Preprocessing.
# 2. Pre-training of the base transformer model.
# 3. Fine-tuning & Alignment (including RLHF when applicable).
# 4. Validation & Safety Testing.
# 5. Deployment & Inference (APIs, web interfaces).
# 6. Monitoring & Updates (feedback, retraining).

# -----------------------------
# PART 6: SAFE CITY PROJECTS (HIGH-LEVEL)
# -----------------------------

# Typical Components
# - Sensor / Data Capture Layer: CCTV cameras, IoT sensors, traffic sensors, environmental sensors.
# - Data Transport & Network Infrastructure: High-speed networks, edge computing for low latency.
# - Processing / Analytics Layer: Real-time analytics (object detection, facial recognition, LPR).
# - Decision Support & Visualization: Dashboards, alerts, map overlays for control centers.
# - Data Storage & Databases: Time-series data, video archives, GIS systems.
# - Privacy & Ethics: Anonymization, access control, cybersecurity, legal compliance.

# -----------------------------
# PART 7: DRONE ARCHITECTURE (PHOTOGRAPHY, WILDLIFE, SURVEY)
# -----------------------------

# Use-case priorities differ (weddings: smooth footage; wildlife: long flight & quiet; survey: precise GPS/RTK).
#
# Hardware components:
# - Frame/Airframe
# - Motors & Propellers
# - ESCs (Electronic Speed Controllers)
# - Battery & Power distribution
# - Sensors (IMU, GPS/GNSS, Barometer, Magnetometer, Cameras, LiDAR/ToF)
# - Camera + Gimbal
# - Flight Controller / Onboard Computer
# - Communication (RC link, video downlink, telemetry)
# - Ground Station / App
#
# Software components:
# - Low-level control loops & stabilization (fast loops maintaining attitude)
# - Sensor fusion / state estimation (IMU + GPS + other sensors)
# - Mission / navigation module (waypoints, trajectory)
# - Camera control & payload management (gimbal control, camera settings)
# - Communication & telemetry (status, video feed)
# - Safety logic (return-to-home, obstacle avoidance, geofencing)
#
# Example stack: PX4 flight controller + companion computer (NVIDIA Jetson/RPi) for vision/AI tasks.

# -----------------------------
# PART 8: CUSTOM FULL-STACK AI DEVELOPER CURRICULUM (ROADMAP)
# -----------------------------

# Overview:
# - Domains: ML/DL, Web Development, Cybersecurity, Robotics
# - Phased learning with projects that connect domains.
#
# Phase 1 — Foundations (3–4 months)
# - Courses: Python, Math for ML, Git, Andrew Ng's ML course
# - Projects: Data visualizer, linear/logistic regression from scratch, CLI app
#
# Phase 2 — ML & DL (4–5 months)
# - Courses: Deep Learning Specialization, CNNs, NLP + Transformers, Generative AI
# - Projects: MNIST classifier, image classifier, sentiment classifier, mini-chatbot
#
# Phase 3 — Web Development (3–4 months)
# - Courses: Front-end (React), Back-end (Django/Flask), APIs, Deployment (AWS/DevOps)
# - Projects: Flask/Django app with authentication, deploy ML model as REST API, dashboard
#
# Phase 4 — Cybersecurity (3 months)
# - Courses: IBM Cybersecurity Analyst, OWASP, AI Security/Adversarial ML
# - Projects: Secure login system, vulnerability scan, adversarial testing fix
#
# Phase 5 — Robotics & Edge AI (4–6 months)
# - Courses: Robotics Specialization (Penn), Perception for Self-Driving Cars, TinyML, ROS
# - Projects: Simulate drone flight path (Gazebo+ROS), line-following robot, TinyML face or motion detector
#
# Phase 6 — Capstone (2–3 months)
# - Project ideas:
#   1. AI-Powered Surveillance Drone: Object detection + flight control + web dashboard + secure comms
#   2. Smart City Monitoring Web App: Sensors + ML event detection + secure admin UI
#   3. AI-Based Healthcare Dashboard: Predict anomalies + secure deployed model + cloud monitoring

# Estimated timeline: ~18–24 months for comprehensive coverage (self-paced).

# -----------------------------
# PART 9: PRACTICAL SUGGESTIONS & NEXT STEPS
# -----------------------------

# - When building, prefer project-based learning: each course should produce a tangible project.
# - Integrate security from the start: secure authentication, encryption, validated inputs.
# - Use version control and CI/CD pipelines for reproducibility and deployment experience.
# - Start small with robotics: use simulation (Gazebo, Webots) before buying hardware.
# - For LLMs: experiment with open-source models and smaller checkpoints before using large-hosted APIs.
# - Document every project: README, data sources, model card, and security considerations.

# -----------------------------
# PART 10: SUPERVISED, UNSUPERVISED, AND REINFORCEMENT LEARNING
# -----------------------------

# Supervised Learning
# - Basic definition:
#   Supervised learning trains models on labeled data, where inputs are paired with correct outputs, to predict or classify new data.
# - Small working principle:
#   The model learns mappings from inputs to outputs by minimizing prediction errors on training data.
# - Flow of work:
#   1. Collect labeled dataset (features + targets).
#   2. Train model to minimize loss between predictions and labels.
#   3. Validate on unseen data.
#   4. Deploy for predictions on new inputs.

# Unsupervised Learning
# - Basic definition:
#   Unsupervised learning discovers patterns in unlabeled data, such as clusters or associations, without explicit guidance.
# - Small working principle:
#   It uses similarity metrics to group data or reduce dimensions, revealing hidden structures.
# - Flow of work:
#   1. Input unlabeled dataset.
#   2. Apply algorithm to find patterns (e.g., clustering).
#   3. Interpret results (e.g., group similarities).
#   4. Use for data exploration or preprocessing.

# Reinforcement Learning (RL)
# - Basic definition:
#   RL trains agents to make sequential decisions by interacting with an environment, maximizing cumulative rewards.
# - Small working principle:
#   The agent learns a policy through trial-and-error, using rewards/penalties to update actions.
# - Flow of work:
#   1. Agent observes state, takes action.
#   2. Environment responds with new state and reward.
#   3. Update value/policy based on reward (e.g., Q-learning).
#   4. Repeat episodes to optimize long-term reward.

# Semi-Supervised Learning
# - Basic definition:
#   Semi-supervised learning uses a small amount of labeled data combined with a large amount of unlabeled data to improve model performance.
# - Small working principle:
#   It leverages unlabeled data to learn better representations, often through pseudo-labeling or consistency regularization.
# - Flow of work:
#   1. Train on labeled data initially.
#   2. Use model to label unlabeled data (pseudo-labels).
#   3. Retrain on combined labeled + pseudo-labeled data.
#   4. Iterate to refine predictions.

# Self-Supervised Learning
# - Basic definition:
#   Self-supervised learning generates labels from the input data itself for pre-training, often used in large models like LLMs.
# - Small working principle:
#   Tasks like predicting masked parts or next tokens create supervisory signals from unstructured data.
# - Flow of work:
#   1. Preprocess data to create proxy tasks (e.g., mask tokens).
#   2. Train model to solve these tasks.
#   3. Fine-tune on downstream labeled tasks.
#   4. Achieves state-of-the-art with less labeled data.

# -----------------------------
# PART 11: CLASSICAL ML ALGORITHMS
# -----------------------------

# Linear Regression
# - Basic definition:
#   Linear Regression models the relationship between inputs and a continuous output using a linear equation.
# - Small working principle:
#   It finds the best-fit line by minimizing the sum of squared errors between predictions and actual values.
# - Flow of work:
#   1. Assume y = mx + b (slope m, intercept b).
#   2. Use least squares or GD to optimize m and b.
#   3. Predict new y for given x.
#   4. Evaluate with metrics like MSE or R-squared.

# Logistic Regression
# - Basic definition:
#   Logistic Regression is used for binary classification, modeling probabilities with a sigmoid function.
# - Small working principle:
#   It transforms linear outputs to probabilities between 0 and 1, using cross-entropy loss.
# - Flow of work:
#   1. Compute linear combination: z = wx + b.
#   2. Apply sigmoid: p = 1 / (1 + e^-z).
#   3. Optimize w and b to minimize loss.
#   4. Classify based on probability threshold (e.g., 0.5).

# K-Nearest Neighbors (KNN)
# - Basic definition:
#   KNN is a non-parametric algorithm that classifies or regresses based on the majority vote or average of the K closest training examples.
# - Small working principle:
#   It uses distance metrics (e.g., Euclidean) to find neighbors, assuming similar points have similar labels.
# - Flow of work:
#   1. Choose K and distance metric.
#   2. For a new point, find K nearest neighbors in training data.
#   3. For classification: majority vote; for regression: average.
#   4. Predict the output.

# Support Vector Machines (SVM)
# - Basic definition:
#   SVM finds the optimal hyperplane that maximizes the margin between classes in classification tasks.
# - Small working principle:
#   It uses support vectors (closest points) and kernels for non-linear boundaries, minimizing misclassifications.
# - Flow of work:
#   1. Map data to higher dimensions if needed (kernel trick).
#   2. Find hyperplane maximizing margin.
#   3. Use soft margins for noisy data (C parameter).
#   4. Classify new points based on side of hyperplane.

# Decision Trees
# - Basic definition:
#   Decision Trees split data recursively based on features to create a tree-like model for classification or regression.
# - Small working principle:
#   It selects splits that maximize information gain or reduce impurity (e.g., Gini, entropy).
# - Flow of work:
#   1. Start at root, choose best feature to split.
#   2. Recurse on subsets until stopping criteria (depth, purity).
#   3. Prune to avoid overfitting.
#   4. Traverse tree for predictions.

# Naive Bayes
# - Basic definition:
#   Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming feature independence.
# - Small working principle:
#   It calculates posterior probabilities using prior and likelihood, efficient for text classification.
# - Flow of work:
#   1. Compute priors P(class) and likelihoods P(feature|class).
#   2. For new data, compute P(class|features) ∝ P(class) * ∏ P(feature|class).
#   3. Choose class with highest posterior.
#   4. Handle zeros with smoothing (e.g., Laplace).

# -----------------------------
# PART 12: ENSEMBLE METHODS
# -----------------------------

# Ensemble Learning (General)
# - Basic definition:
#   Ensemble methods combine multiple models to improve performance, reducing variance or bias.
# - Small working principle:
#   By aggregating predictions (voting, averaging), ensembles are more robust than single models.
# - Flow of work:
#   1. Train base models on data subsets or variations.
#   2. Combine outputs (e.g., majority vote).
#   3. Evaluate ensemble performance.
#   4. Used in bagging, boosting, stacking.

# Bagging (Bootstrap Aggregating)
# - Basic definition:
#   Bagging trains multiple models on bootstrapped subsets and aggregates predictions to reduce variance.
# - Small working principle:
#   Random sampling with replacement creates diverse models; averaging smooths errors.
# - Flow of work:
#   1. Create multiple bootstrapped datasets.
#   2. Train a base model (e.g., tree) on each.
#   3. Aggregate: average for regression, vote for classification.
#   4. Example: Random Forests use bagging + feature randomness.

# Boosting
# - Basic definition:
#   Boosting sequentially trains models, focusing on errors of previous ones, to reduce bias.
# - Small working principle:
#   Each model corrects predecessors by weighting misclassified samples higher.
# - Flow of work:
#   1. Train initial model on data.
#   2. Increase weights of misclassified samples.
#   3. Train next model on weighted data.
#   4. Combine with weighted voting (e.g., AdaBoost, Gradient Boosting).

# Random Forest
# - Basic definition:
#   Random Forest is an ensemble of decision trees using bagging and random feature selection.
# - Small working principle:
#   Diversity from random subsets reduces overfitting; aggregation improves accuracy.
# - Flow of work:
#   1. Bootstrap samples and select random features per split.
#   2. Build multiple trees.
#   3. Aggregate predictions.
#   4. Provides feature importance.

# Gradient Boosting Machines (GBM)
# - Basic definition:
#   GBM builds trees sequentially, each fitting residuals of the previous ensemble.
# - Small working principle:
#   It minimizes loss by adding weak learners, using gradients for corrections.
# - Flow of work:
#   1. Start with initial prediction (mean).
#   2. Compute residuals (errors).
#   3. Fit a tree to residuals.
#   4. Update ensemble: add shrunk tree prediction; repeat.

# XGBoost
# - Basic definition:
#   XGBoost is an optimized GBM implementation with regularization and parallel processing.
# - Small working principle:
#   It adds L1/L2 regularization, handles missing data, and uses second-order gradients for faster convergence.
# - Flow of work:
#   1. Define objective with loss + regularization.
#   2. Build trees using approximate splitting.
#   3. Shrinkage and subsampling for stability.
#   4. Predict with weighted sum of trees.

# Stacking
# - Basic definition:
#   Stacking combines diverse base models' predictions as inputs to a meta-model.
# - Small working principle:
#   Base models provide features; meta-model learns optimal combination.
# - Flow of work:
#   1. Train base models on data.
#   2. Use their predictions as new features.
#   3. Train meta-model on these features.
#   4. Final prediction from meta-model.

# -----------------------------
# PART 13: CLUSTERING AND DIMENSIONALITY REDUCTION
# -----------------------------

# K-Means Clustering
# - Basic definition:
#   K-Means partitions data into K clusters by minimizing intra-cluster variance.
# - Small working principle:
#   It iteratively assigns points to nearest centroids and updates centroids.
# - Flow of work:
#   1. Initialize K centroids randomly.
#   2. Assign each point to closest centroid.
#   3. Update centroids as mean of assigned points.
#   4. Repeat until convergence.

# Hierarchical Clustering
# - Basic definition:
#   Hierarchical Clustering builds a tree of clusters by merging or splitting based on similarity.
# - Small working principle:
#   Agglomerative (bottom-up) merges closest clusters; divisive (top-down) splits.
# - Flow of work:
#   1. Start with each point as a cluster.
#   2. Merge closest pairs (using linkage: single, complete, average).
#   3. Repeat until one cluster or desired level.
#   4. Cut dendrogram for K clusters.

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# - Basic definition:
#   DBSCAN groups dense regions separated by sparse areas, handling noise and arbitrary shapes.
# - Small working principle:
#   It uses epsilon (distance) and minPts to define core points and expand clusters.
# - Flow of work:
#   1. For each point, find neighbors within epsilon.
#   2. If >= minPts, it's core; expand by adding neighbors.
#   3. Border points connect to cores; others are noise.
#   4. Form clusters from connected components.

# Principal Component Analysis (PCA)
# - Basic definition:
#   PCA reduces dimensions by projecting data onto principal components that capture maximum variance.
# - Small working principle:
#   It computes eigenvectors of covariance matrix; top ones form new axes.
# - Flow of work:
#   1. Standardize data.
#   2. Compute covariance matrix.
#   3. Find eigenvalues/eigenvectors; sort descending.
#   4. Project data onto top K components.

# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# - Basic definition:
#   t-SNE visualizes high-dimensional data in low dimensions (2D/3D) by preserving local similarities.
# - Small working principle:
#   It minimizes divergence between high-D similarities and low-D distances using gradients.
# - Flow of work:
#   1. Compute pairwise similarities in high-D (Gaussian).
#   2. Initialize low-D points randomly.
#   3. Optimize low-D (t-distribution) to match using KL divergence.
#   4. Iterate for visualization.

# Autoencoders (for Dimensionality Reduction)
# - Basic definition:
#   Autoencoders are neural networks that learn compressed representations by encoding and decoding data.
# - Small working principle:
#   Encoder compresses to latent space; decoder reconstructs; minimize reconstruction loss.
# - Flow of work:
#   1. Input data to encoder → latent vector.
#   2. Decode latent → reconstructed output.
#   3. Train to minimize difference (e.g., MSE).
#   4. Use latent as reduced features.

# -----------------------------
# PART 14: CONVOLUTIONAL NEURAL NETWORKS (CNNS)
# -----------------------------

# Convolutional Neural Networks (CNNs)
# - Basic definition:
#   CNNs are deep networks for grid-like data (images), using convolutions to extract hierarchical features.
# - Small working principle:
#   Filters slide over inputs to detect patterns; pooling reduces dimensions.
# - Flow of work:
#   1. Input image → convolution layers (filters + activation).
#   2. Pooling (max/avg) for downsampling.
#   3. Flatten and feed to dense layers.
#   4. Output classification/regression.

# Convolution Operation
# - Basic definition:
#   Convolution applies a kernel to input to produce feature maps, detecting local patterns.
# - Small working principle:
#   Kernel slides, multiplies element-wise, sums for each position.
# - Flow of work:
#   1. Define kernel (e.g., 3x3 weights).
#   2. Overlay on input patch → dot product.
#   3. Stride to next position.
#   4. Pad edges if needed.

# Pooling Layers
# - Basic definition:
#   Pooling reduces spatial dimensions, making models invariant to small translations.
# - Small working principle:
#   Max pooling takes max in window; average takes mean.
# - Flow of work:
#   1. Define window size and stride.
#   2. Slide over feature map.
#   3. Compute max/avg per channel.
#   4. Output smaller map.

# Common Architectures (e.g., ResNet, VGG)
# - Basic definition:
#   ResNet uses residual blocks with skip connections to train very deep networks.
#   VGG stacks simple convolutions for depth.
# - Small working principle:
#   Skip connections mitigate vanishing gradients.
# - Flow of work (ResNet):
#   1. Input to conv block.
#   2. Add shortcut (identity) to output.
#   3. Stack blocks for depth.
#   4. Final classification.

# -----------------------------
# PART 15: RECURRENT NEURAL NETWORKS (RNNS) AND LSTMS
# -----------------------------

# Recurrent Neural Networks (RNNs)
# - Basic definition:
#   RNNs process sequences by maintaining hidden states across time steps.
# - Small working principle:
#   Each step uses current input and previous hidden state.
# - Flow of work:
#   1. Initialize hidden state.
#   2. For each time step: hidden = tanh(W_input * input + W_hidden * prev_hidden + b).
#   3. Output from hidden.
#   4. Backprop through time (BPTT).

# Long Short-Term Memory (LSTM)
# - Basic definition:
#   LSTM is an RNN variant with gates to handle long dependencies, avoiding vanishing gradients.
# - Small working principle:
#   Forget, input, output gates control cell state flow.
# - Flow of work:
#   1. Forget gate: decide what to discard from cell.
#   2. Input gate: add new info to cell.
#   3. Update cell state.
#   4. Output gate: produce hidden from cell.

# Gated Recurrent Unit (GRU)
# - Basic definition:
#   GRU is a simplified LSTM with update and reset gates, fewer parameters.
# - Small working principle:
#   Combines forget/input into update; reset for relevance.
# - Flow of work:
#   1. Update gate: how much past to keep.
#   2. Reset gate: ignore past for candidate.
#   3. Compute new hidden.
#   4. Blend old and new hidden.

# Bidirectional RNNs
# - Basic definition:
#   Bidirectional RNNs process sequences forward and backward, capturing full context.
# - Small working principle:
#   Two RNNs: one forward, one reverse; concatenate outputs.
# - Flow of work:
#   1. Run forward RNN on sequence.
#   2. Run backward RNN on reversed sequence.
#   3. Combine hidden states.
#   4. Use for tasks like NER or speech.

# -----------------------------
# PART 16: GENERATIVE MODELS
# -----------------------------

# Generative Adversarial Networks (GANs)
# - Basic definition:
#   GANs consist of a generator creating fake data and a discriminator distinguishing real/fake, trained adversarially.
# - Small working principle:
#   Generator fools discriminator; discriminator improves detection; equilibrium yields realistic data.
# - Flow of work:
#   1. Generator: noise → fake sample.
#   2. Discriminator: classify real/fake.
#   3. Update discriminator on real + fake.
#   4. Update generator to maximize fooling.

# Variational Autoencoders (VAEs)
# - Basic definition:
#   VAEs are probabilistic autoencoders that learn latent distributions for generation.
# - Small working principle:
#   Encoder to mean/variance; sample latent; decoder reconstructs; KL + reconstruction loss.
# - Flow of work:
#   1. Encoder: input → mu, sigma.
#   2. Sample z ~ N(mu, sigma).
#   3. Decoder: z → output.
#   4. Minimize ELBO (evidence lower bound).

# Diffusion Models
# - Basic definition:
#   Diffusion Models generate data by reversing a noising process, iteratively denoising.
# - Small working principle:
#   Forward: add noise; reverse: predict and remove noise.
# - Flow of work:
#   1. Train to predict noise at each step.
#   2. Start from pure noise.
#   3. Iteratively denoise over T steps.
#   4. Output clean sample.

# Normalizing Flows
# - Basic definition:
#   Normalizing Flows transform simple distributions to complex ones via invertible functions.
# - Small working principle:
#   Chain bijective transformations; exact likelihood computation.
# - Flow of work:
#   1. Start with base distribution (e.g., Gaussian).
#   2. Apply invertible flows (e.g., affine).
#   3. Compute log-likelihood for training.
#   4. Sample by inverting flows.

# -----------------------------
# PART 17: ADVANCED TOPICS
# -----------------------------

# Transfer Learning
# - Basic definition:
#   Transfer Learning reuses pre-trained models on new tasks, fine-tuning for domain adaptation.
# - Small working principle:
#   Freeze base layers, train new head; saves time/data.
# - Flow of work:
#   1. Load pre-trained model (e.g., ImageNet CNN).
#   2. Replace output layer for new classes.
#   3. Fine-tune with small learning rate.
#   4. Evaluate on target task.

# Federated Learning
# - Basic definition:
#   Federated Learning trains models across decentralized devices without sharing raw data.
# - Small working principle:
#   Devices train locally; aggregate updates centrally.
# - Flow of work:
#   1. Send global model to devices.
#   2. Local training on private data.
#   3. Upload gradients/updates.
#   4. Average and update global model.

# Explainable AI (XAI)
# - Basic definition:
#   XAI provides interpretable insights into model decisions, building trust.
# - Small working principle:
#   Methods like SHAP, LIME attribute importance to features.
# - Flow of work:
#   1. Train black-box model.
#   2. Use explainer (e.g., LIME perturbs inputs).
#   3. Generate local/global explanations.
#   4. Visualize feature contributions.

# Quantum Neural Networks (QNNs)
# - Basic definition:
#   QNNs leverage quantum computing for neural networks, using qubits and quantum gates.
# - Small working principle:
#   Quantum circuits parameterize layers; superposition/entanglement for efficiency.
# - Flow of work:
#   1. Encode data into quantum states.
#   2. Apply parameterized quantum circuits.
#   3. Measure outputs.
#   4. Optimize parameters classically.

# Bayesian Neural Networks
# - Basic definition:
#   BNNs place distributions over weights, providing uncertainty estimates.
# - Small working principle:
#   Use variational inference to approximate posteriors.
# - Flow of work:
#   1. Define priors on weights.
#   2. Infer posteriors given data.
#   3. Sample weights for predictions.
#   4. Compute uncertainty (variance).

# Time Series Forecasting (e.g., ARIMA, Prophet)
# - Basic definition:
#   Time series models predict future values based on past patterns, handling trends/seasonality.
# - Small working principle:
#   ARIMA: autoregressive + differencing + moving average.
# - Flow of work (ARIMA):
#   1. Make series stationary (differencing).
#   2. Fit AR(p) and MA(q) terms.
#   3. Forecast future steps.
#   4. Evaluate with AIC or residuals.

# AutoML (Automated Machine Learning)
# - Basic definition:
#   AutoML automates model selection, hyperparameter tuning, and feature engineering.
# - Small working principle:
#   Uses search algorithms (e.g., Bayesian optimization) over pipelines.
# - Flow of work:
#   1. Define search space (models, params).
#   2. Evaluate candidates on validation.
#   3. Select best pipeline.
#   4. Deploy optimized model.

# Active Learning
# - Basic definition:
#   Active Learning iteratively selects the most informative data points for labeling to minimize labeling costs while maximizing model performance.
# - Small working principle:
#   Query strategies like uncertainty sampling choose samples where the model is least confident.
# - Flow of work:
#   1. Train initial model on small labeled set.
#   2. Query unlabeled data for high-value samples (e.g., entropy-based).
#   3. Label queried samples and retrain.
#   4. Repeat until performance plateaus.

# -----------------------------
# PART 18: MODERN LLM TECHNIQUES AND PROMPT ENGINEERING
# -----------------------------

# Prompt Engineering
# - Basic definition:
#   Prompt Engineering is the practice of designing effective inputs (prompts) for LLMs to elicit desired outputs.
# - Small working principle:
#   Well-crafted prompts guide the model's reasoning, reducing ambiguity and improving accuracy.
# - Flow of work:
#   1. Define task and desired output format.
#   2. Craft prompt with instructions, examples, or context.
#   3. Test and iterate on prompt variations.
#   4. Deploy in applications for consistent results.

# Chain of Thought (CoT) Prompting
# - Basic definition:
#   Chain of Thought prompting encourages LLMs to break down problems into step-by-step reasoning before answering.
# - Small working principle:
#   By including "Let's think step by step" or examples with intermediate steps, the model simulates human-like reasoning.
# - Flow of work:
#   1. Provide prompt with reasoning examples or instructions.
#   2. Model generates intermediate thoughts.
#   3. Use thoughts to arrive at final answer.
#   4. Improves performance on complex tasks like math or logic.

# Few-Shot Learning
# - Basic definition:
#   Few-Shot Learning adapts models to new tasks with only a few examples provided in the prompt.
# - Small working principle:
#   LLMs generalize from in-context examples without weight updates.
# - Flow of work:
#   1. Include 1-5 input-output examples in prompt.
#   2. Add new input for inference.
#   3. Model predicts based on patterns in examples.
#   4. Effective for tasks with limited data.

# Zero-Shot Learning
# - Basic definition:
#   Zero-Shot Learning performs tasks without any task-specific examples, relying on pre-trained knowledge.
# - Small working principle:
#   Prompt describes the task directly; model infers from general understanding.
# - Flow of work:
#   1. Craft descriptive prompt (e.g., "Classify this text as positive or negative:").
#   2. Input data.
#   3. Model outputs based on zero examples.
#   4. Useful for broad, unseen tasks.

# Retrieval-Augmented Generation (RAG)
# - Basic definition:
#   RAG enhances LLMs by retrieving relevant external knowledge and incorporating it into generation.
# - Small working principle:
#   Query a knowledge base (e.g., vector DB), retrieve docs, and condition generation on them.
# - Flow of work:
#   1. Embed query and search for similar docs.
#   2. Retrieve top-k relevant passages.
#   3. Augment prompt with retrieved content.
#   4. Generate response grounded in facts.

# Mixture of Experts (MoE)
# - Basic definition:
#   MoE architectures route inputs to specialized "expert" sub-networks for efficient scaling.
# - Small working principle:
#   A gating network selects experts per token; only active experts compute, saving resources.
# - Flow of work:
#   1. Input to gating network.
#   2. Gate selects top-k experts.
#   3. Experts process and combine outputs.
#   4. Scales to trillions of parameters (e.g., GPT-4 uses MoE variants).

# Parameter-Efficient Fine-Tuning (PEFT)
# - Basic definition:
#   PEFT fine-tunes large models by updating only a small subset of parameters, reducing compute.
# - Small working principle:
#   Methods like adapters or prefixes add tunable modules while freezing base model.
# - Flow of work:
#   1. Load pre-trained model.
#   2. Insert PEFT modules (e.g., LoRA adapters).
#   3. Train only new parameters on task data.
#   4. Merge for inference.

# Low-Rank Adaptation (LoRA)
# - Basic definition:
#   LoRA is a PEFT method that injects low-rank matrices into layers for efficient fine-tuning.
# - Small working principle:
#   Decompose updates as low-rank: ΔW = A * B, where A and B are small.
# - Flow of work:
#   1. Add LoRA adapters to target layers.
#   2. Train A and B while freezing W.
#   3. During inference, add ΔW to W.
#   4. Reduces VRAM and training time.

# QLoRA (Quantized LoRA)
# - Basic definition:
#   QLoRA combines quantization with LoRA for even more efficient fine-tuning on limited hardware.
# - Small working principle:
#   Quantize model to 4-bit, apply LoRA, and use double quantization for gradients.
# - Flow of work:
#   1. Quantize base model (e.g., to NF4).
#   2. Apply LoRA adapters.
#   3. Fine-tune with quantized gradients.
#   4. Dequantize for high-quality inference.

# -----------------------------
# PART 19: GRAPH NEURAL NETWORKS AND SPECIALIZED ARCHITECTURES
# -----------------------------

# Graph Neural Networks (GNNs)
# - Basic definition:
#   GNNs process graph-structured data by propagating information across nodes and edges.
# - Small working principle:
#   Each layer aggregates neighbor features, updating node representations.
# - Flow of work:
#   1. Represent graph with nodes, edges, features.
#   2. Message passing: aggregate from neighbors.
#   3. Update node states with activation.
#   4. Predict node/graph-level tasks.

# Graph Convolutional Networks (GCNs)
# - Basic definition:
#   GCNs apply convolutions on graphs, normalizing adjacency for spectral or spatial aggregation.
# - Small working principle:
#   Layer: H^(l+1) = σ(Â H^l W^l), where Â is normalized adjacency.
# - Flow of work:
#   1. Compute normalized adjacency matrix.
#   2. Multiply by current features and weights.
#   3. Apply activation.
#   4. Stack layers for deeper representations.

# Graph Attention Networks (GATs)
# - Basic definition:
#   GATs use attention to weigh neighbor importance in graph aggregation.
# - Small working principle:
#   Compute attention coefficients for neighbors, weighted sum features.
# - Flow of work:
#   1. Compute attention scores using queries/keys.
#   2. Softmax for weights.
#   3. Aggregate weighted neighbor features.
#   4. Multi-head for stability.

# Multimodal Models
# - Basic definition:
#   Multimodal models process and integrate data from multiple modalities (e.g., text + image).
# - Small working principle:
#   Fuse embeddings from modality-specific encoders (e.g., CLIP: contrastive learning).
# - Flow of work:
#   1. Encode each modality (e.g., ViT for images, BERT for text).
#   2. Align/fuse in joint space.
#   3. Perform tasks like VQA or generation.
#   4. Examples: CLIP, DALL-E, GPT-4V.

# Vision Transformers (ViT)
# - Basic definition:
#   ViT applies Transformers to images by treating patches as tokens.
# - Small working principle:
#   Patch embeddings + positionals; self-attention captures global relations.
# - Flow of work:
#   1. Split image into patches, flatten, embed.
#   2. Add class token and positionals.
#   3. Transformer layers process.
#   4. Classify from class token.

# -----------------------------
# PART 20: META-LEARNING AND ADVANCED LEARNING PARADIGMS
# -----------------------------

# Meta-Learning (Learning to Learn)
# - Basic definition:
#   Meta-Learning trains models to adapt quickly to new tasks with few examples.
# - Small working principle:
#   Optimize for fast adaptation across meta-tasks (e.g., MAML: gradient-based).
# - Flow of work:
#   1. Sample meta-tasks from distribution.
#   2. Inner loop: adapt on support set.
#   3. Outer loop: optimize meta-parameters on query set.
#   4. At test: adapt to new task.

# Model-Agnostic Meta-Learning (MAML)
# - Basic definition:
#   MAML finds initial parameters that allow quick fine-tuning on new tasks.
# - Small working principle:
#   Compute gradients of adapted losses for meta-update.
# - Flow of work:
#   1. Initialize θ.
#   2. For each task: θ' = θ - α ∇L_support.
#   3. Meta-loss: sum L_query(θ').
#   4. Update θ -= β ∇meta-loss.

# Continual Learning
# - Basic definition:
#   Continual Learning enables models to learn new tasks sequentially without forgetting old ones.
# - Small working principle:
#   Techniques like replay, regularization prevent catastrophic forgetting.
# - Flow of work:
#   1. Train on task 1.
#   2. For new task: regularize changes to important weights (EWC).
#   3. Replay old samples or generate them.
#   4. Adapt without overwriting prior knowledge.

# Domain Adaptation
# - Basic definition:
#   Domain Adaptation aligns source and target domains for better transfer.
# - Small working principle:
#   Minimize domain discrepancy (e.g., adversarial training).
# - Flow of work:
#   1. Train on source domain.
#   2. Align features (e.g., via DANN: domain classifier).
#   3. Fine-tune on limited target data.
#   4. Improve generalization across domains.

# Neuro-Symbolic AI
# - Basic definition:
#   Neuro-Symbolic AI combines neural networks with symbolic reasoning for interpretable, robust AI.
# - Small working principle:
#   Neural for perception, symbolic for logic/rules.
# - Flow of work:
#   1. Neural module extracts features/symbols.
#   2. Symbolic module applies rules/reasoning.
#   3. Integrate for end-to-end learning.
#   4. Used in planning, QA with logic.

# End of notes.