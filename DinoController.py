import time
import pyautogui
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('Savitar.keras')

# Image Parameters
img_height = 128
img_width = 72

# Take a screenshot and preprocess it
def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = screenshot.resize((img_width, img_height))
    screenshot_array = img_to_array(screenshot) / 255.0
    screenshot_array = np.expand_dims(screenshot_array, axis=0)
    return screenshot_array

# Press the appropriate key based on the model's prediction
def press_key(prediction):
    action = np.argmax(prediction)
    if action == 1:
        pyautogui.press('up')
    elif action == 2:
        pyautogui.press('down')


# Start timer
print("Starting in 5 seconds...")
time.sleep(5)
print("Started!")

# Main loop
try:
    while True:
        screenshot = take_screenshot()
        prediction = model.predict(screenshot)
        press_key(prediction)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopped by user")