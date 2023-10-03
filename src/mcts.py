import torch
import numpy as np

from .gameboard import GameBoard


class Node:
    def __init__(self, state, player, win, legal_moves=None, parent=None, prior=0.0):
        """
        Initialize a new node with the given state, parent, and prior probability.

        Args:
        - state (numpy.ndarray): Encoded game board state, shape (3, 6, 7).
        - parent (Node): Parent node.
        - player (int): The player who's turn it is to play
        - win (bool): If the state is a win. Means the player of the parent node won.
        - prior (float): Prior probability of reaching this node.
        """
        self.state = state
        self.player = player
        self.win = win
        self.legal_moves = legal_moves
        self.parent = parent
        self.prior = prior

        self.children = {}  # Maps actions to child nodes
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        """Check if all possible moves have been expanded at the node."""
        return len(self.children) == len(self.legal_moves)

    def best_child(self, exploration_weight=1.0):
        """
        Return the most promising child node, considering both exploitation and exploration.

        Args:
        - exploration_weight (float): The weight given to the exploration term. Higher values encourage more exploration.

        Returns:
        - Node: The selected child node.
        """
        best_value = -float("inf")
        best_node = None

        for child in self.children.values():
            # The formula for UCT (Upper Confidence Bound applied to Trees)
            uct_value = child.value / (
                1 + child.visits
            ) + exploration_weight * child.prior * np.sqrt(self.visits) / (
                1 + child.visits
            )

            if uct_value > best_value:
                best_value = uct_value
                best_node = child

        return best_node

    @staticmethod
    def print_node_values(node, depth=0):
        """
        Recursively print the values of a node and all its children.

        Args:
        - node (Node): The root node to start traversal from.
        - depth (int): The current depth of the tree. Used for indentation.
        """
        indent = "  " * depth  # Create an indentation based on the depth
        print(
            f"{indent}Value: {node.value}, Visits: {node.visits}, Prior: {node.prior}"
        )

        for child in node.children.values():
            print_node_values(child, depth + 1)


class MCTS:
    def __init__(self, env, neural_network, device, exploration_weight=1.0):
        """
        Args:
        - neural_network (torch.nn.Module): The neural network for policy and value estimation
        - exploration_weight (float): The weight for the exploration term in the UCT formula.
        """
        self.env = env
        self.neural_network = neural_network
        self.device = device
        self.exploration_weight = exploration_weight
        self.root = Node(
            state=self.env.encode_board(),
            player=self.env.current_player,
            win=False,
            legal_moves=np.arange(7),
        )

    def make_move(self, action):
        """
        Update the root of the MCTS tree to the node corresponding to the chosen action.

        Args:
        - action (int): The action that was taken.
        """
        self.root = self.root.children[action]
        self.root.parent = None

    def search(self, num_simulations):
        """
        Perform a fixed number of MCTS simulations to find the best move.

        Args:
        - state (numpy.ndarray): The root state to start the search from.
        - num_simulations (int): The number of MCTS simulations to run.

        Returns:
        - int: The action (move) that corresponds to the most visited child of the root.
        """

        for _ in range(num_simulations):
            # Selection: Select a leaf node
            leaf_node = self.select_node()

            # Expansion: Generate children from legal moves and move probabilities
            state_tensor = (
                torch.FloatTensor(leaf_node.state).unsqueeze(0).to(self.device)
            )
            prior_probs, value_estimate = self.neural_network(state_tensor)
            prior_probs = prior_probs.cpu().detach().numpy()
            value_estimate = value_estimate.cpu().detach().numpy()

            if leaf_node.win:
                self.backpropagate(leaf_node, value_estimate)
                continue

            prior_probs = prior_probs[0]  # NOTE: only working with batch of 1 currently

            self.expand(leaf_node, prior_probs)

            # Simulation: Estimate the value of the new board state
            value_estimate = self.simulate(leaf_node)

            # Backpropagation: Update the value and visits of the nodes
            self.backpropagate(leaf_node, value_estimate)

    def select_node(self):
        """
        Select the most promising leaf node according to the UCT formula.

        Returns:
        - Node: The selected leaf node.
        """
        node = self.root
        while node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)

        return node

    def expand(self, leaf_node, prior_probs):
        """
        Expand a leaf node by generating child nodes for all possible actions.

        Args:
        - leaf_node (Node): The node to expand.
        - prior_probs (numpy.ndarray): The prior probabilities for each action, as given by the neural network.
        """
        epsilon = 0.25
        alpha = 0.5

        dirichlet_noise = np.random.dirichlet([alpha] * len(leaf_node.legal_moves))

        for i, move in enumerate(leaf_node.legal_moves):
            adjusted_prior = (1 - epsilon) * prior_probs[
                move
            ] + epsilon * dirichlet_noise[i]
            state, player = self.env.decode_state(leaf_node.state)
            next_state, win = self.env.next_state(move, state, player)
            encoded_next_state = self.env.encode_board(next_state, (3 - player))
            legal_moves = self.env.generate_legal_moves(next_state)

            child_node = Node(
                state=encoded_next_state,
                player=(3 - player),
                win=win,
                legal_moves=legal_moves,
                parent=leaf_node,
                prior=adjusted_prior,
            )
            leaf_node.children[move] = child_node

    def simulate(self, node):
        """
        Simulate the outcome of a node using the neural network's value head.

        Args:
        - node (Node): The node to simulate.

        Returns:
        - float: The estimated value of the node's state.
        """
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
        _, value = self.neural_network(state_tensor)
        return value.cpu().detach().numpy()

    def backpropagate(self, node, value):
        """
        Backpropagate the value up the tree to update the nodes' statistics.

        Args:
        - node (Node): The starting node to backpropagate from.
        - value (float): The value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            if node.player == 2:
                node.value += 1 * value.item()
            elif node.player == 1:
                node.value += -1 * value.item()
            node = node.parent

    def get_policy(self, root, temperature=1):
        """
        Get the policy from the root node based on the visit counts of the child nodes.

        Args:
        - root (Node): The root node of the MCTS tree.
        - temp (float): The temperature parameter to control exploration.

        Returns:
        - numpy.ndarray: The policy as a probability distribution over actions.
        """
        actions = list(root.children.keys())
        visit_counts = np.array([root.children[a].visits for a in actions])

        if temperature == 0:
            policy = np.zeros(7, dtype=np.float32)
            policy[actions[np.argmax(visit_counts)]] = 1
            return policy

        policy = np.zeros(7, dtype=np.float32)
        visit_counts_temp = visit_counts ** (1 / temperature)
        policy[actions] = visit_counts_temp / np.sum(visit_counts_temp)

        return policy / policy.sum()
