from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


class Game:
    def __init__(self):

        self.init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas';"
        self.chrome_driver_path = "./chromedriver"
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=self.chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.get('https://elgoog.im/t-rex/')
        self._driver.execute_script(self.init_script)
        self.press_up()
        self.toggle_bot()

    def get_driver(self):
        return self._driver

    def toggle_bot(self):
        self._driver.find_element_by_id("botStatus").click()

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def end(self):
        self._driver.close()
