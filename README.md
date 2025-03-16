# Deep-Learning-Project
Various deep learning models and techniques, including the LMS Algorithm for adaptive filtering, a simple neural network, CIFAR-10 classification, optimized training pipelines, advanced CNN architectures, RNNs for sentiment analysis, and a Large Language Model for text generation. 
# Least Mean Squares (LMS) Algorithm
I implemented the Least Mean Squares (LMS) Algorithm using NumPy to train a linear regression model. Your goal is to analyze how different learning rates affect the LMS algorithm's performance and compare it to the least squares (Wiener) solution.
First, I computed the optimal weight using the least squares (Wiener) solution. This involves calculating the closed-form solution for the weight vector and evaluating the Mean Squared Error (MSE) of the dataset using this optimal weight.

Next, I implemented the LMS algorithm, where the model is trained sequentially, updating weights using one data point at a time.initialize the weight vector and use a learning rate of r= 0.005. The training will run for 20 epochs, and you will track the MSE loss across epochs to observe the learning process.
After training, I visualized the dataset and the learned models by plotting a 3D scatter plot of all data points. You will also overlay the linear models obtained from both the least squares and LMS-trained weights to compare their fits.

Finally, analyze how the learning rate (r) affects training by repeating the LMS process with different values of r (0.01, 0.05, 0.1, and 0.5). and track the MSE loss for each case and interpret the impact of learning rate choices. Additionally, you will test an extreme case with r = 1 to observe its effect on training stability.

This lab will helping understand both the theoretical and practical aspects of the LMS algorithm, including optimization, convergence behavior, and the impact of hyperparameter tuning.

# Simple NN
I implement and analyze a Simple Neural Network (Simple NN) model using PyTorch. The model consists of three convolutional layers, three max-pooling layers, and two fully connected (FC) layers. Your main tasks involve defining the neural network, analyzing the shape of each layer, and computing key metrics such as the number of parameters and Multiply-Accumulate Operations (MACs).

First, I modify the SimpleNN model by replacing the nn.Conv2d and nn.Linear layers with customized PyTorch classes. This ensures that the implementation aligns with PyTorch’s built-in structures while maintaining flexibility. After modifying the model, I need to verify that it runs correctly by passing a sample input without errors.

Next, I analyze the input and output shapes of each layer, including the feature maps and weight tensors for convolutional and fully connected layers. Using a loop, I extract and print these shapes, then compute the total number of parameters and MACs in each layer. These results will be recorded in Table 2 for further analysis.

Additionally, an optional bonus task involves plotting histograms of weight elements and gradients, performing a backward pass, and comparing gradient distributions when weights are set to zero. This will help understanding weight initialization effects and training dynamics.

# SimpleNN for CIFAR-10 classification + update more robust version
In this project, I implemented and trained a Simple Neural Network (SimpleNN) for CIFAR-10 classification using PyTorch. The model consists of two convolutional layers, two max-pooling layers, and three fully connected (FC) layers. My main tasks included verifying the model's implementation, preparing the dataset, setting up the training pipeline, and analyzing the training performance.

First, I checked the model's implementation by passing dummy inputs and verifying the output shapes to ensure correctness. Then, I processed the CIFAR-10 dataset by applying preprocessing steps such as normalization and transformation to prepare the images for training. Instead of using the standard torchvision.datasets, I used a custom CIFAR-10 dataset class from tools.dataset.

Next, I set up an efficient data loading pipeline using torch.utils.data.DataLoader and deployed the model on a GPU for faster training. To ensure the model was running on the GPU, I used the nvidia-smi command. After that, I defined the loss function and optimizer, which are crucial for training, and configured the training loop with the given hyperparameters.

During training, I monitored training and validation accuracy to evaluate model performance. Finally, I experimented with learning rate decay to observe its impact on accuracy compared to a fixed learning rate

Then, I improved the training pipeline for CIFAR-10 classification by implementing data augmentation, modifying the model architecture, and tuning hyperparameters. The goal was to enhance model performance and achieve at least 70% validation accuracy.

First, I applied data augmentation techniques, including random cropping with padding and random flipping, to help prevent overfitting. I then compared the validation accuracy of the model trained with and without augmentation.

Next, I modified the SimpleNN model design by adding batch normalization (BN) layers after each convolutional layer to stabilize training and allow a larger learning rate. I then empirically evaluated how batch normalization affects accuracy and learning rate. Additionally, I replaced all ReLU activations with Swish and observed whether Swish improved performance.

Finally, I tuned hyperparameters to further optimize the model. I experimented with different learning rates (from 1.0 to 0.001) to analyze their impact on training performance. I also tested different L2 regularization strengths to see how weight decay affects accuracy. Additionally, I switched from L2 to L1 regularization, manually adding L1 loss to the training function, and compared the distribution of weight parameters between L1 and L2 regularization.

Through these improvements, I developed a more robust training pipeline for CIFAR-10 classification.

# Advanced CNN architectures
In this project, I improved the CIFAR-10 classification model by implementing an advanced CNN architecture, ResNet-20, which has a much higher learning capacity compared to the previously used SimpleNN model. The goal was to enhance model performance and achieve over 90% validation accuracy.

First, I implemented the ResNet-20 architecture following the guidelines from the original ResNet paper. I built the deep neural network (DNN) model from scratch without copying any online code to fully understand its implementation. ResNet-20 uses residual connections, which help prevent vanishing gradients and improve training stability.

Next, I tuned the ResNet-20 model using techniques such as data augmentation, learning rate decay, and hyperparameter optimization. The model was trained for up to 200 epochs, allowing it to achieve a validation accuracy above 90%. To ensure training progress was not lost, I regularly saved the trained model during training.

Finally, after training, I tested the model on unseen data and generated predictions. I saved these predictions in a predictions.csv file using the provided script. The predictions were compared to the ground-truth labels for final evaluation. The completed resnet-cifar10.ipynb notebook and predictions.csv were submitted for grading.

# Recurrent Neural Network for Sentiment Analysis

In this project, I implemented a Recurrent Neural Network (RNN) with LSTM for sentiment analysis using the IMDB dataset, which consists of 50,000 movie reviews labeled as either positive or negative. The goal was to process textual data, build an LSTM-based model, and train it to classify sentiment.

First, I implemented a data loader function to read the dataset from the local disk and split it into training, validation, and test sets in a 7:1:2 ratio. Then, I built a vocabulary from the training corpus by computing word frequencies, filtering out stop words, and retaining only words with a minimum frequency.

Next, I created a tokenization function that converts words into their corresponding indices based on the vocabulary. I also implemented the __getitem__ function for the dataset class, which retrieves a review and its label, tokenizes the text, truncates or pads it to a fixed length, and converts labels into binary indices (positive = 1, negative = 0).

For model implementation, I built an LSTM model consisting of an embedding layer, LSTM layer, linear layer, and dropout layer. To handle variable-length sequences, I padded sequences to ensure uniform input sizes. After implementing the model, I trained it for five epochs and monitored training progress and validation accuracy. Finally, I analyzed the model's performance and predictions.

# Large Langugae Model for Text Generation

In this project, I worked with GPT-2, a pre-trained large language model (LLM), using Hugging Face Transformers to generate and fine-tune text. The tasks included loading the model, preparing data, running inference, and fine-tuning using different methods.

First, I loaded the pre-trained GPT-2 model and used it to generate text. The .generate() function was used for text generation, and .decode() converted generated tokens into readable text. Then, I prepared the wiki-text dataset, using only 10% of the data for practice due to its large size.

Next, I evaluated GPT-2’s inference performance using perplexity, a common metric for language models. I then fine-tuned GPT-2 on the wiki-text dataset, monitoring training and validation loss across epochs. After fine-tuning, I evaluated the perplexity of the fine-tuned model and generated text to compare results.

Additionally, I implemented LoRA (Low-Rank Adaptation) fine-tuning using Hugging Face’s peft library, which enables efficient adaptation of large models. Finally, I compared the training time and performance of full fine-tuning vs. LoRA fine-tuning, analyzing their differences in perplexity and generated text quality. This project helped me understand fine-tuning techniques and efficiency optimizations for LLMs.
