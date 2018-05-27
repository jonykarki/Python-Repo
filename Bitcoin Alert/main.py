from win10toast import ToastNotifier
from pygame import mixer
import schedule
import requests
import json

API_URL = "https://api.coindesk.com/v1/bpi/currentprice/{}.json"
PRICE_INCREASED = "Bitcoin Price Increased!"
PRICE_DECREASED = "Bitcoin Price Decreased!"
CURRENT_PRICE = 0.0
CURRENCY = "USD"
TOASTER = ToastNotifier()

def get_current_price():
    """Returns current bitcoin price for a
    certain currency"""
    response = requests.get(API_URL.format(CURRENCY))
    res = json.loads(response.text)
    rate = float(res["bpi"][CURRENCY]["rate_float"])
    return rate

# notify the user about the changes
def show_notification(toast_title, toast_message, toast_duration=10):
    TOASTER.show_toast(
        title=toast_title,
        msg=toast_message,
        duration=toast_duration,
        icon_path="bitcoin.ico"
    )

def play_sound():
    """Plays the sound when price changes"""
    mixer.init()
    mixer.music.load('sound.mp3')
    mixer.music.play()

def check_current_price():
    """Check the current rate of bitcoin"""
    rate = get_current_price()
    global CURRENT_PRICE
    if rate < CURRENT_PRICE:
        play_sound()
        msg = "Current Price is: {}".format(rate)
        show_notification(PRICE_DECREASED,msg)
        CURRENT_PRICE = rate
    elif rate > CURRENT_PRICE:
        msg = "Current Price is: {}".format(rate)
        show_notification(PRICE_INCREASED,msg)
        CURRENT_PRICE = rate
    else:
        msg = "Surprise!! No change in price"
        show_notification("NO CHANGE", msg,5)

if __name__ == "__main__":
    CURRENT_PRICE = get_current_price()
    schedule.every(1).minute.do(check_current_price)
    while True:
        schedule.run_pending()