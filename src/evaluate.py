import numpy as np

from src.mcts import MCTS


def evaluate_nn(env, nn1, nn2, device, num_games=100):
    nn1_wins = 0
    nn2_wins = 0
    draws = 0

    for _ in range(num_games):
        env.reset()
        game_over = False
        mcts1 = MCTS(env, nn1, device)
        mcts2 = MCTS(env, nn2, device)

        while not game_over:
            mcts1.search(num_simulations=50)
            policy1 = mcts1.get_policy(mcts1.root, temperature=0)

            mcts2.search(num_simulations=50)
            policy2 = mcts2.get_policy(mcts2.root, temperature=0)

            if env.current_player == 1:
                chosen_move = np.argmax(policy1)
            else:
                chosen_move = np.argmax(policy2)

            env.make_move(chosen_move)

            # Update both MCTS roots to the new state
            mcts1.make_move(chosen_move)
            mcts2.make_move(chosen_move)

            if env.is_game_over:
                game_over = True
                if env.winner == 1:
                    nn1_wins += 1
                elif env.winner == 2:
                    nn2_wins += 1
                else:
                    draws += 1
    return nn1_wins / num_games
