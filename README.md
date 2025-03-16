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

# SimpleNN for CIFAR-10 classification
In this project, I implemented and trained a Simple Neural Network (SimpleNN) for CIFAR-10 classification using PyTorch. The model consists of two convolutional layers, two max-pooling layers, and three fully connected (FC) layers. My main tasks included verifying the model's implementation, preparing the dataset, setting up the training pipeline, and analyzing the training performance.

First, I checked the model's implementation by passing dummy inputs and verifying the output shapes to ensure correctness. Then, I processed the CIFAR-10 dataset by applying preprocessing steps such as normalization and transformation to prepare the images for training. Instead of using the standard torchvision.datasets, I used a custom CIFAR-10 dataset class from tools.dataset.

Next, I set up an efficient data loading pipeline using torch.utils.data.DataLoader and deployed the model on a GPU for faster training. To ensure the model was running on the GPU, I used the nvidia-smi command. After that, I defined the loss function and optimizer, which are crucial for training, and configured the training loop with the given hyperparameters.

During training, I monitored training and validation accuracy to evaluate model performance. Finally, I experimented with learning rate decay to observe its impact on accuracy compared to a fixed learning rate

Then, I improved the training pipeline for CIFAR-10 classification by implementing data augmentation, modifying the model architecture, and tuning hyperparameters. The goal was to enhance model performance and achieve at least 70% validation accuracy.

First, I applied data augmentation techniques, including random cropping with padding and random flipping, to help prevent overfitting. I then compared the validation accuracy of the model trained with and without augmentation.

Next, I modified the SimpleNN model design by adding batch normalization (BN) layers after each convolutional layer to stabilize training and allow a larger learning rate. I then empirically evaluated how batch normalization affects accuracy and learning rate. Additionally, I replaced all ReLU activations with Swish and observed whether Swish improved performance.

Finally, I tuned hyperparameters to further optimize the model. I experimented with different learning rates (from 1.0 to 0.001) to analyze their impact on training performance. I also tested different L2 regularization strengths to see how weight decay affects accuracy. Additionally, I switched from L2 to L1 regularization, manually adding L1 loss to the training function, and compared the distribution of weight parameters between L1 and L2 regularization.

Through these improvements, I developed a more robust training pipeline for CIFAR-10 classification.
