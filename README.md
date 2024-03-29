# Project README

## Autonomus Lab 3: HPC and Deep Learning

This project explores the intersection of High-Performance Computing (HPC) and Deep Learning, with a focus on optimization algorithms and strategies. The primary objectives include experimenting with different learning rates and optimization methods, analyzing their impact on convergence, and evaluating their performance on both a simple linear model and the MNIST dataset.

### Project Structure:

- **Exercise 1: Optimization Algorithms for Linear Model**
  - Gradient Descent: Experimentation with different learning rates, visualizing convergence.
  - Momentum Optimizer: Introduction of momentum to accelerate learning, analysis of its impact.
  - Adam Optimizer: Adaptive learning rates and momentum for effective optimization.
  - Comparison: Evaluating and comparing the performance of the three optimization algorithms.

- **Exercise 2: Optimization Algorithms for MNIST Dataset**
  - Applying optimization algorithms to the more complex MNIST dataset.
  - Comparative analysis of Gradient Descent, Momentum Optimizer, and Adam Optimizer.
  - Exploration of convergence rates, optimal learning rates, and their impact on performance.

- **Exercise 3: Performance Improvement**
  - Modification of code to train for more epochs and use validation data.
  - Achieving higher accuracy on the MNIST test set.
  - Configuration details: Learning rate (1e-4), Batch Size (50), Epochs (30).

- **Exercise 4: Multi-GPU Usage**
  - Implementation of code for utilizing multiple GPUs.
  - Analysis of performance improvements, including the impact of epochs and batch size.
  - Comparative evaluation with different GPU configurations.

### How to Use:

1. **Dependencies:**
   - All dependencies for each exercise are included in the respective `launch_ex*.sh` scripts.

2. **Experiments:**
   - Navigate to each exercise folder for detailed scripts and results.
   - Experiment with different configurations and parameters as needed.

3. **Code Modifications:**
   - The code is structured to allow easy adaptation for different datasets and model architectures.

4. **Results and Visualizations:**
   - Explore the generated figures and visualizations in each exercise folder to gain insights into convergence and performance.

### Conclusion:

This project provides a comprehensive exploration of optimization algorithms in the context of deep learning, offering practical examples and insights. The findings are applicable to a range of scenarios and can guide users in selecting appropriate algorithms and parameters for their specific tasks. The combination of HPC and deep learning optimization strategies opens avenues for enhanced model training efficiency.