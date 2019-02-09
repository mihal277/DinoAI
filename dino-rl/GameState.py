class GameState:
    def __init__(self, agent, game, game_wrapper):
        self._agent = agent
        self._game = game
        self._game_wrapper = game_wrapper
        self._display = game_wrapper.show_img() #display the processed image on screen using openCV, implemented using python coroutine
        self._display.__next__() # initiliaze the display coroutine

    def get_state(self, actions):
        self._game_wrapper.actions_df.loc[len(self._game_wrapper.actions_df)] = actions[1] # storing actions in a dataframe
        score = self._game.get_score()
        reward = 0.01
        is_over = False #game over
        if actions[1] == 1:
            reward = -5
            self._agent.jump()
        image = self._game_wrapper.grab_screen(self._game._driver)
        self._display.send(image) #display the image on screen
        if self._agent.is_crashed():
            self._game_wrapper.scores_df.loc[len(self._game_wrapper.scores_df)] = score # log the score when game is over
            self._game.restart()
            reward = -100
            is_over = True
        return image, reward, is_over #return the Experience tuple
