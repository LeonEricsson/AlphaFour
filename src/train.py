import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.mcts import MCTS


class C4Dataset(Dataset):
    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        s, p, v = self.training_data[idx]
        return torch.FloatTensor(s), torch.FloatTensor(p), torch.FloatTensor([v])


def train_nn(nn, training_data, optimizer, criterion, device, epochs=10, batch_size=32):
    nn.train()
    dataset = C4Dataset(training_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for _, (states, policies, values) in enumerate(train_loader):
            states, policies, values = (
                states.to(device),
                policies.to(device),
                values.to(device),
            )

            optimizer.zero_grad()

            predicted_policies, predicted_values = nn(states)
            loss = criterion(
                predicted_values.view(-1), values.view(-1), predicted_policies, policies
            )

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed.", end="\r")
    print()


def MCTS_self_play(env, nn, device, num_games, num_simulations):
    training_data = []

    for _ in range(num_games):
        env.reset()
        game_data = []
        game_over = False
        move_count = 0

        mcts = MCTS(env, nn, device)

        while not game_over:
            mcts.search(num_simulations=num_simulations)

            policy = mcts.get_policy(
                mcts.root, temperature=(1 if move_count < 10 else 0.1)
            )

            # Save current state and policy, value is assumed to be 0 to reduce computation
            current_state = env.encode_board()
            game_data.append((current_state, policy, 0))

            # Perform action in environment and prune mcts search tree
            chosen_move = np.random.choice(range(7), p=policy)
            env.make_move(chosen_move)
            mcts.make_move(chosen_move)
            move_count += 1

            # Check for game over
            if env.is_game_over:
                game_over = True
                if env.winner == 1:
                    game_data = [(s, p, 1) for s, p, _ in game_data]
                elif env.winner == 2:
                    game_data = [(s, p, -1) for s, p, _ in game_data]

        training_data.extend(game_data)

    return training_data
