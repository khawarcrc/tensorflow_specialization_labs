# Lab 1: Convolutional Neural Network (CNN) for Image Classification
"""
This notebook demonstrates how to:
- Load and explore an image dataset (e.g., CIFAR-10, MNIST, or custom)
- Normalize and preprocess images for training
- Visualize sample images with labels
- Build a CNN pipeline (Conv2D -> MaxPooling -> Flatten -> Dense)
- Explain convolution, filters, and feature maps with examples
- Add dropout and batch normalization for regularization
- Compile the model with an optimizer (Adam) and loss function
- Train the CNN and track accuracy/loss with validation data
- Plot training/validation performance curves
- Evaluate the model on test data
- Inspect learned filters and feature maps
- Test predictions on unseen images
- Save and reload the trained model
- Discuss limitations (overfitting, dataset size, computational needs)
- Suggest improvements (transfer learning, data augmentation, deeper networks)
"""

# Lab 2: Speech / Audio Classification (Keyword Spotting / Simple Speech Recognition)
"""
This notebook demonstrates how to:
- Load a small audio dataset (e.g., Speech Commands dataset)
- Visualize audio waveforms
- Convert raw audio to spectrograms / MFCCs
- Normalize features for neural network training
- Split into training/validation/test sets
- Build an audio classification pipeline (Conv2D -> LSTM/GRU -> Dense)
- Explain why spectrograms work well with CNNs
- Train the model on labeled audio samples
- Plot accuracy and loss curves
- Evaluate test performance and confusion matrix
- Test the model on custom audio recordings
- Explore overfitting vs. generalization issues
- Save spectrogram preprocessing pipeline and trained model
- Discuss limitations (noise, accents, small vocabulary)
- Suggest next steps (RNNs, Transformers for speech, larger vocab)
"""

# Lab 3: Recommendation System (Collaborative Filtering + Neural Networks)
"""
This notebook demonstrates how to:
- Load a user–item rating dataset (e.g., MovieLens)
- Explore rating distribution and sparsity
- Preprocess users, items, and ratings into numeric IDs
- Build embeddings for users and items
- Create a neural recommendation model (Embedding -> Dot product / Dense)
- Train the model with observed ratings
- Evaluate with RMSE / MAE metrics
- Visualize training and validation error curves
- Generate recommendations for a specific user
- Explore item similarity via cosine distance in embedding space
- Inspect learned user and item embeddings
- Handle cold-start problems (new users/items)
- Save embeddings and recommendation model
- Discuss collaborative vs. content-based approaches
- Suggest improvements (hybrid models, sequence-aware recommenders, Transformers)
"""

# Lab 4: Sentiment Analysis with Transformers (Text Classification)
"""
This notebook demonstrates how to:
- Load a text dataset (e.g., IMDB reviews)
- Explore dataset statistics (review length, positive/negative balance)
- Preprocess text using a tokenizer (WordPiece/BPE)
- Convert text into token IDs with padding
- Build a classification pipeline (Embedding -> Transformer Encoder -> Dense)
- Explain self-attention and positional encoding
- Compile the model with Adam optimizer and cross-entropy loss
- Train with training/validation splits
- Plot training vs. validation accuracy/loss
- Evaluate on test set with classification report
- Visualize attention weights for sample sentences
- Test predictions on custom text samples
- Save tokenizer and trained model
- Discuss limitations (bias, compute cost, interpretability)
- Suggest next steps (fine-tuning pre-trained BERT, multi-class sentiment)
"""


# Lab 1: Fine-Tuning a Pre-trained Large Language Model (LLM) with LoRA/PEFT
"""
This notebook demonstrates how to:
- Load a pre-trained model (e.g., GPT-2, LLaMA-2, Mistral) from Hugging Face
- Tokenize and preprocess domain-specific data (e.g., legal, finance, medical text)
- Understand parameter-efficient fine-tuning (LoRA, PEFT)
- Attach LoRA adapters to attention layers
- Train the model with limited compute using transformers + peft
- Monitor loss curves and evaluate perplexity
- Compare full fine-tuning vs. LoRA in terms of parameters and speed
- Generate domain-adapted text samples
- Evaluate outputs with BLEU/ROUGE or human feedback
- Save and reload the fine-tuned adapters
- Test zero-shot vs fine-tuned performance
- Explore catastrophic forgetting issues
- Deploy fine-tuned model with inference pipeline
- Discuss limitations (data size, hallucinations)
- Suggest next steps (RLHF, instruction tuning, retrieval-augmented fine-tuning)
"""

# Lab 2: Vision-Language Model (VLM) – Image Captioning with CLIP + Transformer
"""
This notebook demonstrates how to:
- Load a pre-trained vision encoder (e.g., CLIP, ViT)
- Load a text decoder (Transformer / GPT-2)
- Preprocess dataset (e.g., Flickr8k or MS-COCO captions)
- Extract image embeddings with CLIP
- Align embeddings with text tokens
- Build a training pipeline (ImageEncoder -> TransformerDecoder)
- Train on paired image-caption data
- Monitor loss and generate captions during training
- Evaluate with BLEU, CIDEr, METEOR scores
- Visualize attention maps (words ↔ image regions)
- Generate captions for unseen images
- Compare results with pre-trained BLIP or LLaVA
- Save and reload model checkpoints
- Discuss multimodal challenges (compute cost, data size)
- Suggest next steps (instruction-tuned VLMs, grounding with object detection)
"""

# Lab 3: Modern Recommendation System with Embeddings + Retrieval
"""
This notebook demonstrates how to:
- Load an e-commerce or movie dataset (e.g., Amazon, MovieLens)
- Preprocess user–item interactions
- Train embeddings for users and items (Embedding -> DotProduct)
- Introduce Two-Tower models (user tower + item tower)
- Train with sampled softmax / contrastive loss
- Evaluate with Recall@K, NDCG@K metrics
- Build an ANN index (FAISS / ScaNN) for fast retrieval
- Generate top-K recommendations for a user
- Visualize embedding space with t-SNE
- Handle cold-start (content features, side info)
- Compare collaborative vs hybrid approaches
- Deploy retrieval + ranking pipeline
- Test recommendations in an interactive loop
- Save embeddings and deploy with FAISS index
- Discuss next steps (context-aware recsys, sequence models like SASRec, LLM-driven recsys)
"""

# Lab 4: Retrieval-Augmented Generation (RAG) for Domain Q&A
"""
This notebook demonstrates how to:
- Load a base LLM (e.g., GPT-J, LLaMA-2)
- Prepare a domain knowledge corpus (e.g., company docs, PDFs)
- Split text into chunks and embed with sentence-transformers
- Build a vector database (FAISS, Pinecone, Weaviate)
- Implement similarity search to retrieve top-k documents
- Build a RAG pipeline (Retriever -> LLM)
- Generate answers grounded in retrieved passages
- Compare outputs with/without retrieval
- Evaluate with factual accuracy metrics
- Add citation/attribution to answers
- Handle hallucinations with answer verification
- Fine-tune retriever for domain-specific queries
- Deploy as an API or chatbot
- Save embeddings, retriever index, and model
- Discuss limitations (latency, context length, privacy)
"""


# Lab 5: Machine Learning Ops (MLOps) – Model Deployment with FastAPI + Docker
"""
This notebook demonstrates how to:
- Train a simple ML model (e.g., RandomForest, XGBoost, or small neural net)
- Save model artifacts using joblib or torch.save
- Create a FastAPI service for inference
- Write an API endpoint (/predict) that takes JSON input
- Test locally with curl or requests
- Containerize the service using Docker
- Build and run Docker image with the ML API
- Handle environment reproducibility with requirements.txt
- Add logging and request tracking
- Discuss CI/CD for model updates
- Deploy on cloud (AWS, GCP, Azure, or Hugging Face Spaces)
- Monitor latency and throughput
- Handle versioning and rollback of models
- Add authentication (API keys, tokens)
- Discuss limitations (scaling, retraining pipelines)
"""

# Lab 6: Tabular Data – Feature Engineering + XGBoost / LightGBM
"""
This notebook demonstrates how to:
- Load a real-world dataset (e.g., Kaggle Titanic, housing prices)
- Explore data distributions and missing values
- Apply feature engineering (scaling, encoding categorical vars, feature crosses)
- Split into train/validation/test
- Train a baseline model with logistic regression or decision trees
- Introduce gradient boosting methods (XGBoost, LightGBM, CatBoost)
- Tune hyperparameters (learning rate, depth, n_estimators)
- Track performance with ROC AUC, F1, PR curves
- Apply k-fold cross-validation
- Visualize feature importance
- Explore SHAP/interpretability
- Compare boosting vs. deep learning on tabular
- Handle imbalance with SMOTE/weighted loss
- Save and reload model with pickle
- Discuss legacy but high-demand role of boosting models in finance, health, risk
"""

# Lab 7: Time Series Forecasting with LSTMs and Transformers
"""
This notebook demonstrates how to:
- Load a real time-series dataset (stock prices, energy consumption, sales)
- Explore seasonality, trends, autocorrelation
- Apply windowing/sliding techniques for supervised data
- Build a baseline ARIMA/Prophet model
- Prepare sequences for deep learning (LSTM/GRU)
- Build an LSTM forecasting pipeline
- Train and monitor validation error (MAE, RMSE)
- Visualize predictions vs. ground truth
- Introduce a Transformer-based Time Series Model (Informer/TemporalFusion)
- Compare performance of LSTM vs Transformer
- Add external covariates (holidays, promotions)
- Handle multivariate forecasting
- Save trained models and scalers
- Discuss applications (finance, demand forecasting, supply chain)
- Suggest next steps (hybrid models, probabilistic forecasting)
"""

# Lab 8: Anomaly Detection in Streaming Data
"""
This notebook demonstrates how to:
- Load or simulate streaming data (network traffic, IoT sensor logs)
- Explore statistical outliers with z-scores and isolation forests
- Preprocess data with rolling windows
- Train an unsupervised anomaly model (Autoencoder, One-Class SVM)
- Monitor reconstruction error for anomalies
- Build a simple streaming pipeline (Kafka, or simulate with Python generator)
- Perform real-time anomaly scoring
- Visualize anomaly detection results
- Handle concept drift in streaming
- Evaluate precision/recall in anomaly detection
- Compare statistical vs ML-based approaches
- Save anomaly detection model
- Deploy anomaly alerts with API integration
- Discuss applications (fraud detection, predictive maintenance)
- Suggest next steps (transformer-based anomaly detection, federated setups)
"""

# Lab 9: Computer Vision – Transfer Learning with Pre-trained CNNs
"""
This notebook demonstrates how to:
- Load an image dataset (custom or benchmark)
- Normalize and augment images
- Load a pre-trained CNN (ResNet, EfficientNet, MobileNet)
- Freeze base layers and add custom dense head
- Train with transfer learning on new dataset
- Compare training speed vs. from-scratch CNN
- Fine-tune selected layers for better accuracy
- Use early stopping and checkpoint saving
- Evaluate with confusion matrix, precision/recall
- Visualize Grad-CAM saliency maps
- Handle small dataset with heavy augmentation
- Save fine-tuned model
- Test predictions on new images
- Discuss use cases (medical imaging, defect detection)
- Suggest next steps (self-supervised pre-training, multimodal)
"""

# Lab 10: End-to-End Data Pipeline with Airflow + ML Model
"""
This notebook demonstrates how to:
- Install and configure Apache Airflow
- Create DAGs for ML workflow (ETL -> Train -> Deploy)
- Load raw dataset (CSV, database, or API)
- Define preprocessing tasks (cleaning, feature engineering)
- Train ML model as an Airflow task
- Save model artifacts to storage (S3, GCS, MinIO)
- Run evaluation step and log metrics
- Conditional branching based on accuracy thresholds
- Deploy model if threshold is met 
- Schedule retraining pipelines with Airflow
- Monitor DAG execution with UI
- Handle task retries and failures
- Integrate MLflow for experiment tracking
- Discuss real-world MLOps integration
- Suggest next steps (Kubernetes + Airflow, orchestration at scale)
"""
