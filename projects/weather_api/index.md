# Weather API example

This project was to create a weather report application written in Python PEP8 style using the public endpoint https://api.openweathermap.org/data/2.5/.

---

## Motivation

The goal was to create a functional Python script where a user is able to submit the City or zip code and get the weather forecast and relevant information for the forecast. The primary reason is to show the cleanliness of the PEP8 coding style which is highly readable.

---

## Code Style

[PEP8](https://www.python.org/dev/peps/pep-0008

---

## Example Output

```
Welcome to Berkeley's weather forecast retriever


Please enter a zip code or city to get forecast: 60005


Weather forcast retrieved for Arlington Heights:
-----------------------------------------------------------------------------------------------------------------------------
|       Datetime       |  Weather  |  Min. Temp (F)  |  Max Temp (F)  |  Windspeed (MPH)  |  Wind Direction  |  Humidity %  |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-25 21:00:00  |  Clouds   |      46.02      |      46.9      |       10.65       |       ENE        |      69      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 00:00:00  |   Rain    |      42.71      |     43.25      |       12.08       |        NE        |      72      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 03:00:00  |   Rain    |      41.11      |     41.36      |       12.86       |       NNE        |      84      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 06:00:00  |   Rain    |      38.53      |     38.59      |       16.73       |        N         |      87      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 09:00:00  |   Rain    |      36.9       |      36.9      |       12.91       |       NNW        |      81      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 12:00:00  |  Clouds   |      35.02      |     35.02      |        7.9        |       NNW        |      77      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 15:00:00  |  Clouds   |      39.97      |     39.97      |       5.48        |        NW        |      59      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 18:00:00  |  Clouds   |      48.9       |      48.9      |       4.63        |       WNW        |      46      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-26 21:00:00  |  Clouds   |      52.38      |     52.38      |       6.06        |        W         |      45      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 00:00:00  |  Clouds   |      50.13      |     50.13      |       2.86        |       WSW        |      55      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 03:00:00  |  Clouds   |      48.36      |     48.36      |       2.55        |       SSW        |      61      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 06:00:00  |  Clouds   |      46.65      |     46.65      |       3.47        |        SE        |      69      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 09:00:00  |  Clouds   |      45.77      |     45.77      |       3.91        |        SE        |      73      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 12:00:00  |  Clouds   |      44.56      |     44.56      |       6.04        |        SE        |      84      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 15:00:00  |  Clouds   |      52.5       |      52.5      |       11.88       |        S         |      68      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 18:00:00  |  Clouds   |      60.39      |     60.39      |       14.56       |       SSW        |      61      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-27 21:00:00  |  Clouds   |      64.96      |     64.96      |       12.97       |        SW        |      58      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 00:00:00  |  Clouds   |      60.98      |     60.98      |       10.76       |        SW        |      71      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 03:00:00  |   Rain    |      55.85      |     55.85      |       8.55        |       WNW        |      80      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 06:00:00  |  Clouds   |      49.71      |     49.71      |       9.17        |        NW        |      72      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 09:00:00  |  Clouds   |      45.84      |     45.84      |       8.88        |        NW        |      76      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 12:00:00  |   Rain    |      41.14      |     41.14      |       12.5        |       WNW        |      87      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 15:00:00  |   Snow    |      38.77      |     38.77      |       14.36       |        NW        |      67      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 18:00:00  |  Clouds   |      42.03      |     42.03      |       15.5        |        NW        |      50      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-28 21:00:00  |  Clouds   |      46.35      |     46.35      |       15.37       |        NW        |      30      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 00:00:00  |  Clouds   |      43.88      |     43.88      |       9.26        |        NW        |      36      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 03:00:00  |   Clear   |      40.73      |     40.73      |       3.98        |       WNW        |      45      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 06:00:00  |  Clouds   |      39.24      |     39.24      |       3.53        |       WSW        |      50      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 09:00:00  |  Clouds   |      38.88      |     38.88      |       4.21        |       SSW        |      52      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 12:00:00  |  Clouds   |      38.37      |     38.37      |       7.31        |       SSW        |      58      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 15:00:00  |   Clear   |      47.62      |     47.62      |       11.77       |       SSW        |      44      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 18:00:00  |  Clouds   |      56.34      |     56.34      |       15.37       |       SSW        |      39      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-29 21:00:00  |   Clear   |      60.58      |     60.58      |       16.42       |       SSW        |      40      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 00:00:00  |   Clear   |      56.3       |      56.3      |       16.62       |       SSW        |      50      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 03:00:00  |   Clear   |      52.34      |     52.34      |       17.22       |       SSW        |      56      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 06:00:00  |  Clouds   |      49.78      |     49.78      |       16.64       |       SSW        |      58      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 09:00:00  |  Clouds   |      48.02      |     48.02      |       15.26       |       SSW        |      61      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 12:00:00  |  Clouds   |      46.58      |     46.58      |       14.94       |       SSW        |      63      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 15:00:00  |  Clouds   |      53.24      |     53.24      |       16.51       |       SSW        |      50      |
-----------------------------------------------------------------------------------------------------------------------------
| 2021-03-30 18:00:00  |  Clouds   |      62.69      |     62.69      |       17.25       |        SW        |      36      |
-----------------------------------------------------------------------------------------------------------------------------
Would you like to make another request [y/n]:N
Thank you for using Berkeley's weather forcast retriever.Have a good day!
```

---

## How to Use it

Install the [requests](https://pypi.org/project/requests/) Python 3 library.

Then simply run the script directly in the terminal:
```sh
python get_forecast.py

# or if you need to specify version

python3 get_forecast.py
```

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


MIT License

Copyright (c) 2021 Berkeley Willis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.