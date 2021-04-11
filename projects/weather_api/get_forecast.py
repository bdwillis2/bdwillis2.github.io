# File: week12.py
# Name: Berkeley D. Willis
# Date: 11/09/2019
# Course: DSC510-T301 - Introduction to Programming
# Desc:  A simple program that will allow a user to pull and display some
#          general weather information from a location that the user provides by
#          zipcode or city name.
# Usage: A user starts the program intially through the python interpreter and
#          then uses the prompts given to provide zipcodes or city names, and
#          exit when the user enters exit.

import requests

#------------------------------------------------------------------------------
# Function: degToCompass()
#
# Parameter:
#    In:     deg int - The degree value of the direction of the wind
#    Out:    None
#    In/Out: None
#
# Returns:  String that represents the cardinal direction of the wind
# Desc:     Takes degree value of wind direction and create human readable
#             cardinal direction
#------------------------------------------------------------------------------
def degToCompass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[ int((deg/22.5)+.5) % len(dirs) ]

#------------------------------------------------------------------------------
# Function: prettyPrint()
#
# Parameter:
#    In:     weather_json dict - This is the interpreted json that is to be
#              printed in a readable way
#    Out:    None
#    In/Out: None
#
# Returns:  None
# Desc:     Prints the requested weather json object from the request in a
#             readable way
#------------------------------------------------------------------------------
def prettyPrint(weather_json):
    wthr_fmt = max([(len(str(entry['weather'][0]['main']))) for entry in weather_json['list']])
    wthr_fmt = max(9, wthr_fmt)
    min_fmt = max([(len(str(entry['main']['temp_min']))) for entry in weather_json['list']])
    min_fmt = max(15, min_fmt)
    max_fmt = max([(len(str(entry['main']['temp_max']))) for entry in weather_json['list']])
    max_fmt = max(14, max_fmt)
    wndspd_fmt = max([(len(str(entry['wind']['speed']))) for entry in weather_json['list']])
    wndspd_fmt = max(17, wndspd_fmt)
    wndchl_fmt = max([(len(str(entry['wind']['deg']))) for entry in weather_json['list']])
    wndchl_fmt = max(16, wndchl_fmt)
    humid_fmt = max([(len(str(entry['main']['humidity']))) for entry in weather_json['list']])
    humid_fmt = max(12, humid_fmt)
    print("\n\nWeather forcast retrieved for {}:".format(weather_json['city']['name']))
    print("-"*(wthr_fmt+min_fmt+max_fmt+wndspd_fmt+wndchl_fmt+humid_fmt+42))
    print("| {:^{width}} | ".format("Datetime", width=20) +
           "{:^{width}} | ".format("Weather", width=wthr_fmt) +
           "{:^{width}} | ".format("Min. Temp (F)", width=min_fmt) +
           "{:^{width}} | ".format("Max Temp (F)", width=max_fmt) +
           "{:^{width}} | ".format("Windspeed (MPH)", width=wndspd_fmt) +
           "{:^{width}} | ".format("Wind Direction", width=wndchl_fmt) +
           "{:^{width}} |".format("Humidity %", width=humid_fmt)
         )
    print('-'*(wthr_fmt+min_fmt+max_fmt+wndspd_fmt+wndchl_fmt+humid_fmt+42))
    for entry in weather_json['list']:
        print("| {:^{width}} | ".format(entry['dt_txt'], width=20) +
                "{:^{width}} | ".format(entry['weather'][0]['main'], width=wthr_fmt) +
                "{:^{width}} | ".format(entry['main']['temp_min'], width=min_fmt) +
                "{:^{width}} | ".format(entry['main']['temp_max'], width=max_fmt) +
                "{:^{width}} | ".format(entry['wind']['speed'], width=wndspd_fmt) +
                "{:^{width}} | ".format(degToCompass(entry['wind']['deg']), width=wndchl_fmt) +
                "{:^{width}} |".format(entry['main']['humidity'], width=humid_fmt)
             )
        print('-'*(wthr_fmt+min_fmt+max_fmt+wndspd_fmt+wndchl_fmt+humid_fmt+42))

#------------------------------------------------------------------------------
# Function: submitRequest()
#
# Parameter:
#    In:     apiArgs String - Arguments that are used for location data given
#              for the request
#    Out:    None
#    In/Out: None
#
# Returns:  None
# Desc:     Attempts to submit the request to the weather api, and will execute
#             prettyPrint on response if the request was successful.
#------------------------------------------------------------------------------
def submitRequest(apiArgs):
    retrieving = True
    failures = 0
    API_KEY="3f3089a12c580c4c1865c2da159827a8"
    baseURL = "https://api.openweathermap.org/data/2.5/forecast?"
    while retrieving:
        response = requests.get(baseURL+apiArgs+"&APPID="+API_KEY+"&units=imperial")
        try:
            resp_json = response.json()
            if resp_json['cod']!="200":
                failures = failures + 1
            else:
                prettyPrint(resp_json)
                return
            if failures >= 5:
                print("Too many failed attempts to contact the service." +
                       "Please check your connection and try again later")
                return
        except Exception as e:
            print("Exception caught: "+e)
            failures = failures + 1
    print("There was an issue when trying to connect to the weather webservice.")
    print("Please check your connection and try again.")




#------------------------------------------------------------------------------
# Function: main()
#
# Parameter:
#    In:     None
#    Out:    None
#    In/Out: None
#
# Returns:  None
# Desc:     Main function the will execute when the script is used
#------------------------------------------------------------------------------
def main():
    running = True
    print("Welcome to Berkeley's weather forecast retriever")
    while running:
        apiArgs = ""
        userChoice = input("\n\nPlease enter a zip code or city to get forecast: ")
        if userChoice.isdigit():
            apiArgs = "zip={},us".format(userChoice)
        else:
            apiArgs = "q={},us".format(userChoice)
        response = submitRequest(apiArgs)
        gettingInput = True
        userChoice = ""
        while gettingInput:
            userChoice = input("Would you like to make another request [y/n]:")
            if userChoice.lower()=="y" or userChoice.lower()=="yes":
                gettingInput = False
            elif userChoice.lower()=="n" or userChoice.lower()=="no":
                gettingInput = False
                running = False
            else:
                print("Your selection of '{}' was invalid.".format(userChoice))
    print("Thank you for using Berkeley's weather forcast retriever." +
            "Have a good day!")

main()
