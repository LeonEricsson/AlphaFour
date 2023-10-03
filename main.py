import torch
import numpy as np

from src.gameboard import GameBoard
from src.nn import C4Net, C4NetLoss
from src.train import MCTS_self_play, train_nn
from src.evaluate import evaluate_nn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GameBoard()
    nn = C4Net().to(device)
    baseline_nn = C4Net().to(device)
    baseline_nn.load_state_dict(
        nn.state_dict()
    )  # nn = baseline_nn when training from scratch
    optimizer = torch.optim.Adam(nn.parameters())
    criterion = C4NetLoss()

    num_iterations = 1
    win_rate_threshold = 0.55

    # MCTS
    num_self_play_games = 1
    num_simulations = 50

    ## NN
    epochs = 10
    batch_size = 32

    ## Eval
    num_eval_games = 100

    for i in range(num_iterations):
        print(f"Iteration {i + 1}...")

        # Step 1: Generate training data through self-play
        training_data = MCTS_self_play(env, nn, device, num_games, num_simulations)

        # Step 2: Train neural network
        train_nn(nn, training_data, optimizer, criterion, device, epochs, batch_size)

        print("Evaluating")
        # Step 3: Evaluate the neural network
        new_win_rate = evaluate_nn(env, nn, baseline_nn, device, num_eval_games)

        # Step 4: If the new network is better, update it
        if new_win_rate > win_rate_threshold:
            print("New network is better, updating...")
            torch.save(nn.state_dict(), f"../models/model_iteration_{i + 1}.pth")


if __name__ == "__main__":
    main()
