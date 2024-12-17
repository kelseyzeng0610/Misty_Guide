import time
from mistyPy.Robot import Robot
# Using this GitRepo: https://github.com/MistyCommunity/Python-SDK -- in BETA so I have to supplement some behavior myself

import base64
from gtts import gTTS # Used for doing text to speech 
DEBUG_JSON_REQUESTS = False

def JSON_response_to_dictionary(response):
    API_Data = response.json()
    if DEBUG_JSON_REQUESTS:
        for key in API_Data:
            {print(key,":", API_Data[key])}
    return API_Data


if __name__ == "__main__":
    ip_address = "10.5.11.234"
    # Create an instance of a robot
    misty = Robot(ip_address)

    ########################################
    # Debug to make sure connection happened
    misty.MoveArms(30, 20)
    time.sleep(1)
    misty.MoveArms(0, 0)

    # Just some extra debug code just to give y'all some example behaviors
    misty.ChangeLED(red=255, blue=0, green=0) # 0.0 spooky red
    time.sleep(1)
    misty.ChangeLED(red=0, blue=255, green=0) # Let's make it less spooky now :)

    battery_level_response = misty.GetBatteryLevel() # This is just a nice command to know
    battery_level = JSON_response_to_dictionary(battery_level_response)
    print("Battery level: " + str(battery_level['result']['chargePercent']))

    misty.SetBlinking(True)
    misty.DisplayImage("e_Joy2.jpg") 
    time.sleep(2)

    blink_settings_response = misty.GetBlinkSettings() # Want to change blinking and face? Just check out API responses to what is possible
    blink_settings = JSON_response_to_dictionary(blink_settings_response)
    print("Blink Settings: " + str(blink_settings))

    image_list_response = misty.GetImageList()
    image_list = JSON_response_to_dictionary(image_list_response)
    print("Blink Settings: " + str(image_list))

    misty.SetBlinking(False)
    misty.DisplayImage("e_Sleeping.jpg") 
    print()
    print()
    print()
    # I think this is enough to get the vibe of how to use the Python SDK, it's still a little unfinished right now
    # You can read their API for more information on what is possible: https://docs.mistyrobotics.com/misty-ii/web-api/api-reference/#displayimage
    ########################################
    # Okay, now to do text to speech, we'll be doing this the simple and available way!

    #misty.PlayAudio("s_Amazement.wav", volume=100) # This works, so it's not a speaker problem!
    #time.sleep(2)

    text = "Hello World, my name is Misty,and this project is developed by Nikki Falicov and Kelsey Zeng, hope you enjoy it!"
    language = 'en'
    speech = gTTS(text=text, lang=language, slow=False)
    file_name = "audio.mp3"
    speech.save(file_name) # GTTS says you can save as .wav but it lies, it's still a .mp3 object at heart, just a heads up
    
    ENCODING = 'utf-8'
    encode_string = base64.b64encode(open(file_name, "rb").read())
    base64_string = encode_string.decode(ENCODING)

    save_audio_response = misty.SaveAudio(file_name, data=base64_string, overwriteExisting=True, immediatelyApply=True)
    save_audio = JSON_response_to_dictionary(save_audio_response)
    print("Saving Audio Response: " + str(save_audio))
    misty.PlayAudio(file_name, volume=100)