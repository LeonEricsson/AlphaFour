# AlphaFour: Self-Playing Connect Four Agent using AlphaZero Techniques

## Introduction

This repository contains an from-scratch implementation of the AlphaZero algorithm tailored for the game of Connect Four. Neural network guided Monte Carlo Tree Search (MCTS) is used to teach the computer how to play Connect Four. The neural network is trained through self-play, continually improving its ability to evaluate game states and suggest moves. This is an end-to-end solution where the neural network learns the rules, tactics, and strategies of the game without any human intervention.

This project builds an intelligent Connect Four agent using a custom implementation of the AlphaZero algorithm. It implemenets custom PyTorch modules for the neural network components and NumPy for managing the game state and mechanics. The code is written entirely from scratch, allowing for detailed customization. The training loop involves a self-play mechanism, where the agent competes against itself to generate training data, thereby incrementally improving its own performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Structure](#structure)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

1. Clone the repository: `git clone https://github.com/yourusername/AlphaZero-ConnectFour.git`
2. Change to the project directory: `cd AlphaZero-ConnectFour`
3. Install requirements: `pip install -r requirements.txt`

## Iteration Pipeline

The core training logic is based on an iterative approach that involves four key steps. Each iteration aims to improve the agent's capabilities by fine-tuning its underlying neural network.

1. **Self-Play for Data Generation**: The Monte Carlo Tree Search (MCTS) algorithm is employed to simulate games where the agent plays against itself. Each self-play game generates a sequence of board states, policies, and values that serve as training data. The MCTS uses the current neural network for policy and value predictions. 

2. **Training the Neural Network**: The neural network is trained using the newly generated self-play data. The training utilizes PyTorch and is optimized to run on CUDA-enabled GPUs.

3. **Evaluation**: Post-training, the newly updated neural network is evaluated against a baseline network to measure its performance. This is again done using MCTS simulations.

4. **Model Update**: If the new network shows improvement over the baseline network, it becomes the new baseline for the next iteration. The model's parameters are then saved for future use.


## Structure

- `src/`
  - `main.py`: Entry point for training the model.
  - `mcts.py`: Implementation of the Monte Carlo Tree Search (MCTS) algorithm.
  - `nn.py`: Neural network model.
  - `train.py`: Training loop.
  - `gameboard.py`: Connect Four game logic and rules.
  - `evaluate.py`: Script for evaluating the trained model against baseline.

- `models/`: Folder for storing model checkpoints.

- `README.md`: This file.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add some fooBar'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

