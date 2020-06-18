# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import requests
import time
from getpass import getpass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


# %%
driver_path = "/usr/local/bin/chromedriver"

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')


# %%
driver = webdriver.Chrome(executable_path=driver_path,
                          chrome_options=chrome_options
                         )
driver.get(f"https://www.kaggle.com/account/login?phase=emailSignIn&returnUrl=%2F")
login_fields = driver.find_elements_by_xpath("//input[@class='mdc-text-field__input']")
signin_button = driver.find_element_by_xpath("//button")
email = input("Enter email: ")
password = getpass("Password: ")
login_fields[0].send_keys(email)
login_fields[1].send_keys(password)
signin_button.click()
driver.save_screenshot("signed_in.png")


# %%
driver.get(f"https://www.kaggle.com/c/birdsong-recognition/data")
time.sleep(5)
download_all_button = driver.find_elements_by_xpath("//span[@class='mdc-button__label']")
print(len(download_all_button))
# # download_all_button[0].click()
time.sleep(10)
# download_all_button[-1].click()
driver.save_screenshot("compete.png")


# %%

driver.close()

