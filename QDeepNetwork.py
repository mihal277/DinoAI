from collections import deque
import os
import random
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

from DinoGameRunner import FreezingGameRunner

class QNAgent:
    def __init__(self, input_shape, output_shape=3):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._memory = deque(maxlen=5000)
        self._gamma = 0.95  # discount rate
        self._epsilon = 0.5  # exploration rate
        self._epsilon_min = 0.001
        self._epsilon_decay = 0.995
        self._learning_rate = 0.001
        self._model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(kernel_size=2, filters=4, activation='relu', input_shape=self._input_shape))
        model.add(Conv2D(kernel_size=3, filters=8, activation='relu'))
        model.add(Flatten())
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
        self._memory.append((state, action, reward, next_state, done))

    def act(self, state, explore=True):
        if explore and np.random.rand() <= self._epsilon:
            return random.randrange(self._output_shape)
        st = np.reshape(state, (1, state.shape[0], state.shape[1], 1))
        act_values = self._model.predict(st)
        return np.argmax(act_values[0])

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

    def __init__(self, qnagent, game_runner, sample_delay=0.005, initial_delay=2):
        self._qnagent = qnagent
        self._game_runner = game_runner
        self._sample_delay = sample_delay  # how often the screenshots will be taken in sec
        self._initial_delay = initial_delay
        self._scores = []

    def _preprocess(self, img):
        ar = np.array(img)
        ar = np.average(ar[:, :, :3], axis=2)
        ar = ar / 255
        return ar

    def _key_from_actionid(self, action_id):
        action = list(filter(lambda x: x['id'] == action_id, TrainingSupervisor.ACTIONS))[0]
        # print(action['desc'])
        return action['key']

    def train(self, batch_size=32):
        self._game_runner.start(self._initial_delay)

        i = 1
        previous_state = None
        previous_action = -1

        while not self._game_runner.is_crashed():
            new_state = self._preprocess(self._game_runner.take_screenshot())
            new_score = self._game_runner.get_score()

            if previous_action != -1:
                self._qnagent.remember(previous_state, previous_action, new_score, new_state, False)

            action_id = self._qnagent.act(new_state)

            previous_state = new_state
            previous_action = action_id

            self._game_runner.press_key(self._key_from_actionid(action_id))

            if i % 2 * batch_size == 0:
                self._qnagent.replay(batch_size)

            i += 1
            self._game_runner.play(self._sample_delay)


        score = self._game_runner.get_score()
        self._game_runner.exit()

        self._scores.append(score)
        print('Finished game with score of ' + score)

        self._qnagent.remember(previous_state, previous_action, score, None, True)
        self._qnagent.replay(batch_size)


    def get_high_scores(self, count=10):
        return sorted(self._scores)[-count:]

    def get_average_score(self, last_scores=None):
        if last_scores is not None:
            last_scores = min(last_scores, len(self._scores))
            s = self._scores[-last_scores:]
        else:
            s = self._scores
        return sum(s) / len(s)

if __name__ == '__main__':
    working_dir = os.path.dirname(os.path.realpath(__file__))
    t_rex_runner_url = "file://" + working_dir + "/t-rex-runner/index.html"

    qnagent = QNAgent(input_shape=(150,600,1))
    game_runner = FreezingGameRunner(t_rex_runner_url)

    trainer = TrainingSupervisor(qnagent, game_runner)
    trainer.train()
