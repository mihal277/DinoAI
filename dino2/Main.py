from DinoAgent import DinoAgent
from Game import Game
from GameState import GameState
from GameWrapper import GameWrapper
from ModelBuilder import ModelBuilder
from NetworkTrainer import NetworkTrainer

# game parameters
actions = 2  # possible actions: jump, do nothing
gamma = 0.99  # decay rate of past observations original 0.99
observation = 100.  # timesteps to observe before training
explore = 100000  # frames over which to anneal epsilon
final_epsilon = 0.0001  # final value of epsilon
initial_epsilon = 0.1  # starting value of epsilon
replay_memory = 50000  # number of previous transitions to remember
batch = 16  # size of minibatch
frame_per_action = 1
learning_rate = 1e-4
img_rows , img_cols = 80,80
img_channels = 4  # We stack 4 frames

if __name__ == "__main__":
    game_wrapper = GameWrapper()
    game = Game()
    dino_agent = DinoAgent(game)
    game_state = GameState(dino_agent, game, game_wrapper)
    model = ModelBuilder(game_wrapper, img_cols, img_rows, img_channels, actions, learning_rate).model
    network_trainer = NetworkTrainer()
    try:
        network_trainer.train_network(model, game_state, game_wrapper, actions, initial_epsilon, final_epsilon,
                                      learning_rate, observation, replay_memory, frame_per_action, batch, gamma,
                                      explore, observe=False)
    except StopIteration:
        game.end()
