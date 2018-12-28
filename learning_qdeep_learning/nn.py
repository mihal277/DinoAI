from collections import deque
import random

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam
import argparse

from game import Game, Board, Arrows, MoveResult
from supervisor import GameSupervisor

class QNAgent:
    def __init__(self, state_size, action_size):
        self._state_size = state_size
        self._action_size = action_size
        self._memory = deque(maxlen=20000)
        self._gamma = 0.95  # discount rate
        self._epsilon = 1.0  # exploration rate
        self._epsilon_min = 0.1
        self._epsilon_decay = 0.995
        self._learning_rate = 0.001
        self._model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(kernel_size=2, filters=16, activation='relu', input_shape=self._state_size, padding='same'))
        model.add(Conv2D(kernel_size=2, filters=32, activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=self._action_size, activation='softmax'))

        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))
        model.summary()
        return model

    def _learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + self._gamma * np.amax(self._model.predict(next_state)[0]))
        target_f = self._model.predict(state)
        target_f[0][Arrows.to_int(action)] = target
        self._model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))
        # self._learn(state, action, reward, next_state, done)


    def act(self, state, explore=True):
        if explore and np.random.rand() <= self._epsilon:
            return random.randrange(self._action_size)
        act_values = self._model.predict(state)
        return Arrows.from_int(np.argmax(act_values[0]))

    def replay(self, batch_size):
        if batch_size > len(self._memory):
            batch_size = len(self._memory)
        minibatch = random.sample(self._memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self._learn(state, action, reward, next_state, done)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def load(self, name):
        self._model.load_weights(name)

    def save(self, name):
        self._model.save_weights(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_weights', '-o', action='store')
    parser.add_argument('--input_weights', '-i', action='store', required=False)
    parser.add_argument('--episodes', '-e', action='store', type=int, default=1000)
    parser.add_argument('--replay_batch_size', '-r', action='store', default=64)
    parser.add_argument('--max_time', action='store', type=int, default=500)

    args = parser.parse_args()

    agent = QNAgent((9, 9, 1), 4)
    if args.input_weights is not None:
        agent.load(args.input_weights)
    supervisor = GameSupervisor()
    for e in range(args.episodes):
        board = Board(9, 9, 9)
        ending = 'TIMEOUT'

        for time_t in range(args.max_time):
            state = board.get_state().reshape((1, 9, 9, 1))
            move = agent.act(state)

            reward, done, move_result = supervisor.order_move(board, move)
            next_state = board.get_state().reshape((1, 9, 9, 1))

            agent.remember(state, move, reward, next_state, done)
            if done:
                if move_result == MoveResult.HOLE:
                    ending = 'FELL INTO HOLE'
                else:
                    ending = 'ATE ALL THE CHEESE'
                break

        print('episode: {}/{} score: {} {}'.format(e, args.episodes, board.get_score(), ending))
        agent.replay(args.replay_batch_size)

    agent.save(args.output_weights)
