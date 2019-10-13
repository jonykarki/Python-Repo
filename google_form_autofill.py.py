# to help a friend

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver.common.keys import Keys

import time
import random
times = 20
driver = webdriver.Chrome('') #path to chromedriver.exe

try:
    while times:
        driver.get("") #google_form link
        element = driver.find_elements_by_class_name('quantumWizTogglePaperradioEl')
        for i in range(6):
            rand = random.randint(0,1)
            if rand == 1:
                element[i*2].click()
                time.sleep(1)
            elif rand == 0:
                element[(i*2)+1].click()
                time.sleep(1)

        driver.find_element_by_class_name('quantumWizButtonPaperbuttonLabel').click()
        time.sleep(3)
        times-=1

finally:
    driver.quit()