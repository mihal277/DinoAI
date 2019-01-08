from collections import deque
import os
import random
import time

import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

from DinoGameRunner import FreezingGameRunner

class QNAgent:
    def __init__(self, input_shape, output_shape=3, memory_size=2048, error_memory_size=128, max_domination_rate=2):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._memory = deque(maxlen=memory_size)
        self._error_memory = deque(maxlen=error_memory_size)
        self._gamma = 0.95  # discount rate
        self._epsilon = 1  # exploration rate
        self._epsilon_min = 0.001
        self._epsilon_decay = 0.95
        self._exploration_turns = 25
        self._learning_rate = 1e-3
        self._model = self._build_model()

        self._memory_stats = [0] * self._output_shape
        self._error_memory_stats = [0] * self._output_shape
        self._max_domination_rate = max_domination_rate

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(kernel_size=3, filters=16, activation='relu', input_shape=self._input_shape))
        model.add(Conv2D(kernel_size=5, filters=8, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self._output_shape, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))
        model.summary()
        return model

    def _learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_st = np.reshape(next_state, (1, next_state.shape[0], next_state.shape[1], 1))
            target = (reward + self._gamma * np.amax(self._model.predict(next_st)[0]))
        st = np.reshape(state, (1, state.shape[0], state.shape[1], 1))
        target_f = self._model.predict(st)
        target_f[0][action] = target
        self._model.fit(st, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        if done:
            if self._error_memory_stats[action] <= self._max_domination_rate * min(self._error_memory_stats):
                if len(self._error_memory) == self._error_memory.maxlen:
                    self._error_memory_stats[self._error_memory.popleft()[1]] -= 1
                self._error_memory.append((state, action, reward, next_state, done))
                self._error_memory_stats[action] += 1
        else:
            if self._memory_stats[action] <= self._max_domination_rate * min(self._memory_stats):
                if len(self._memory) == self._memory.maxlen:
                    self._memory_stats[self._memory.popleft()[1]] -= 1
                self._memory.append((state, action, reward, next_state, done))
                self._memory_stats[action] += 1

    def explore(self):
        r = random.randrange(self._output_shape)
        print('RANDOM', r)
        return r

    def act(self, state, explore=True):
        if explore and np.random.rand() <= self._epsilon:
            return self.explore()
        st = np.reshape(state, (1, state.shape[0], state.shape[1], 1))
        act_values = self._model.predict(st)
        print(act_values[0], np.argmax(act_values[0]))
        return np.argmax(act_values[0])

    def replay(self, batch_size, error_batch_size):
        if error_batch_size > len(self._error_memory):
            error_batch_size = len(self._error_memory)
        error_minibatch = random.sample(self._error_memory, error_batch_size)
        for state, action, reward, next_state, done in error_minibatch:
            self._learn(state, action, reward, next_state, done)
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

class TrainingSupervisor:
    ACTIONS = [{
        'id': 0,
        'desc': 'RUN',
        'key': ''
    }, {
        'id': 1,
        'desc': 'JUMP',
        'key': 'space'
    }, {
        'id': 2,
        'desc': 'DUCK',
        'key': 'arrow_down'
    }]

    def __init__(self, qnagent, game_runner, sample_delay=0.1, initial_delay=4.75,
                    punish_rate=1):
        self._qnagent = qnagent
        self._game_runner = game_runner
        self._sample_delay = sample_delay  # how often the screenshots will be taken in sec
        self._initial_delay = initial_delay
        self._punish_rate = punish_rate
        self._scores = []

    def _preprocess(self, img):
        img = img.crop((60, 0, 600, 150))
        img = img.resize((135, 50))
        ar = np.array(img)
        ar = np.average(ar[:, :, :3], axis=2)
        ar = ar / 255
        return ar

    def _key_from_actionid(self, action_id):
        action = list(filter(lambda x: x['id'] == action_id, TrainingSupervisor.ACTIONS))[0]
        # print(action['desc'])
        return action['key']

    def train(self, batch_size, error_batch_size, force_explore=False):
        self._game_runner.start(self._initial_delay)

        i = 1
        previous_state = None
        previous_action = -1

        while not self._game_runner.is_crashed():
            new_state = self._preprocess(self._game_runner.take_screenshot())

            if previous_action != -1:
                self._qnagent.remember(previous_state, previous_action, self._game_runner.get_score(), new_state, False)

            if force_explore:
                action_id = self._qnagent.explore()
            else:
                action_id = self._qnagent.act(new_state)

            previous_state = new_state
            previous_action = action_id

            if not force_explore and i % (2 * batch_size) == 0:
                self._qnagent.replay(batch_size, error_batch_size)

            i += 1

            self._game_runner.press_key(self._key_from_actionid(action_id))

            self._game_runner.play(self._sample_delay)


        score = self._game_runner.get_score()
        self._game_runner.exit()

        self._scores.append(score)

        self._qnagent.remember(previous_state, previous_action, self._punish_rate * score, None, True)
        if not force_explore:
            self._qnagent.replay(batch_size, error_batch_size)

        print('\nMemory size:', len(self._qnagent._memory))

        ac = [0, 0, 0]
        for state, action, reward, next_state, done in self._qnagent._memory:
            ac[action] += 1

        print('Memory RUN', ac[0], sep='\t')
        print('Memory JUMP', ac[1], sep='\t')
        print('Memory DUCK', ac[2], sep='\t')
        
        print('\nError memory size:', len(self._qnagent._error_memory))

        ac = [0, 0, 0]
        for state, action, reward, next_state, done in self._qnagent._error_memory:
            ac[action] += 1

        print('Error memory RUN', ac[0], sep='\t')
        print('Error memory JUMP', ac[1], sep='\t')
        print('Error memory DUCK', ac[2], sep='\t')

        return score


    def get_high_scores(self, count=10):
        return sorted(self._scores)[-count:]

    def get_average_score(self, last_scores=None):
        if last_scores is not None:
            last_scores = min(last_scores, len(self._scores))
            s = self._scores[-last_scores:]
        else:
            s = self._scores
        return sum(s) / len(s)
