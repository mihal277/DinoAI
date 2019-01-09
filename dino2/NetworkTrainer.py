import json
import random
import time

import numpy as np
from IPython.display import clear_output
from keras.optimizers import Adam


class NetworkTrainer:
    '''
    main training module
    Parameters:
    * model => Keras Model to be trained
    * game_state => Game State module with access to game environment and dino
    * observe => flag to indicate wherther the model is to be trained(weight updates), else just play
    '''

    def train_network(self, model, game_state, game_wrapper, actions, initial_epsilon, final_epsilon, learning_rate, observation, replay_memory, frame_per_action, batch, gamma, explore, observe=False):
        last_time = time.time()
        # store the previous observations in replay memory
        D = game_wrapper.load_obj("D")  # load from file system
        # get the first state by doing nothing
        do_nothing = np.zeros(actions)
        do_nothing[0] = 1  # 0 => do nothing,
        # 1=> jump

        x_t, r_0, terminal = game_state.get_state(do_nothing)  # get next step after performing the action

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input

        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

        initial_state = s_t

        if observe:
            OBSERVE = 999999999  # We keep observe, never train
            epsilon = final_epsilon
            print("Now we load weight")
            model.load_weights("model.h5")
            adam = Adam(lr=learning_rate)
            model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
        else:  # We go to training mode
            OBSERVE = observation
            epsilon = game_wrapper.load_obj("epsilon")
            model.load_weights("model.h5")
            adam = Adam(lr=learning_rate)
            model.compile(loss='mse', optimizer=adam)

        t = game_wrapper.load_obj("time")  # resume from the previous time step stored in file system
        while (True):  # endless running

            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0  # reward at 4
            a_t = np.zeros([actions])  # action at t

            # choose an action epsilon greedy
            if t % frame_per_action == 0:  # parameter to skip frames for actions
                if random.random() <= epsilon:  # randomly explore an action
                    print("----------Random Action----------")
                    action_index = random.randrange(actions)
                    a_t[action_index] = 1
                else:  # predict the output
                    q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)  # chosing index with maximum q value
                    action_index = max_Q
                    a_t[action_index] = 1  # o=> do nothing, 1=> jump

            # We reduced the epsilon (exploration parameter) gradually
            if epsilon > final_epsilon and t > OBSERVE:
                epsilon -= (initial_epsilon - final_epsilon) / explore

                # run the selected action and observed next state and reward
            x_t1, r_t, terminal = game_state.get_state(a_t)
            print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate
            last_time = time.time()
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3],
                             axis=3)  # append the new image to input stack and remove the first one

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > replay_memory:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:

                # sample a minibatch to train on
                minibatch = random.sample(D, batch)
                inputs = np.zeros((batch, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
                targets = np.zeros((inputs.shape[0], actions))  # 32, 2

                # Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]  # 4D stack of images
                    action_t = minibatch[i][1]  # This is action index
                    reward_t = minibatch[i][2]  # reward at state_t due to action_t
                    state_t1 = minibatch[i][3]  # next state
                    terminal = minibatch[i][4]  # wheather the agent died or survided due the action

                    inputs[i:i + 1] = state_t

                    targets[i] = model.predict(state_t)  # predicted q values
                    Q_sa = model.predict(state_t1)  # predict q values for next step

                    if terminal:
                        targets[i, action_t] = reward_t  # if terminated, only equals reward
                    else:
                        targets[i, action_t] = reward_t + gamma * np.max(Q_sa)

                loss += model.train_on_batch(inputs, targets)
                game_wrapper.loss_df.loc[len(game_wrapper.loss_df)] = loss
                game_wrapper.q_values_df.loc[len(game_wrapper.q_values_df)] = np.max(Q_sa)
            s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
            t = t + 1

            # save progress every 1000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                game_state._game.pause()  # pause game while saving to filesystem
                model.save_weights("model.h5", overwrite=True)
                game_wrapper.save_obj(D, "D")  # saving episodes
                game_wrapper.save_obj(t, "time")  # caching time steps
                game_wrapper.save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
                game_wrapper.loss_df.to_csv("./objects/loss_df.csv", index=False)
                game_wrapper.scores_df.to_csv("./objects/scores_df.csv", index=False)
                game_wrapper.actions_df.to_csv("./objects/actions_df.csv", index=False)
                game_wrapper.q_values_df.to_csv(game_wrapper.q_value_file_path, index=False)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)
                clear_output()
                game_state._game.resume()
            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + explore:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
                  "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")
