from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium import webdriver

class Game:
    def __init__(self):

        self.init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.chrome_driver_path = "./chromedriver"
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=self.chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(self.init_script)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()
