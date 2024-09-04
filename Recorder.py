import keyboard
import time
import pyautogui
import os

# Create the "Images" folder if it doesn't exist
if not os.path.exists('Images'):
    os.makedirs('Images')

def take_screenshot(filename):
    screenshot = pyautogui.screenshot()
    screenshot.save(os.path.join('Images', filename))

# Read keyboard presses and name the .png accordingly
def check_keys_and_screenshot():
    counter = 0
    while True:
        if keyboard.is_pressed('up'):
            filename = f'1_0 {counter}.png' # 1_0 jump
        elif keyboard.is_pressed('down'):
            filename = f'0_1 {counter}.png' # 0_1 crouch
        else:
            filename = f'0_0 {counter}.png' # 0_0 no presses
        
        take_screenshot(filename)
        print(filename)
        counter += 1
        time.sleep(0.273)  # Take a screenshot every 273 milliseconds (Average human reaction time)

        # Check if 'q' is pressed to stop the program
        if keyboard.is_pressed('q'):
            print("Program terminated by user")
            break

# Start timer
print("Starting in 5 seconds...")
time.sleep(5)
print("Started!")

try:
    check_keys_and_screenshot()
except KeyboardInterrupt:
    print("Program terminated")
