# Sensor Strange
This work is a proof of concept for potential side channel attack on smartphones using Deep Learning.

Android does not enforce permission for motion sensors. This work proves that any application in background can infer the text being typed in other applications only from phone movements while typing. This implementaion uses Accelerometer and Gyroscope sensors. From motion sensor readings, we were able to infer 37% of the typed words and were able to detect more than 90% taps on the screen.

## ML pipeline
- Tap Detection - Modified Z score
- Keyboard region classification - RMS distance
- Text inference - Recurrent Neural Network

## Publication
[IJEAT journal paper](https://www.ijeat.org/wp-content/uploads/papers/v9i2/B3432129219.pdf)

## Implementation stack
**Client:** Android  
**Web Server:** Python Flask