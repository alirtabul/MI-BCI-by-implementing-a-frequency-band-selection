/*
This code enables a Serial communication with Python in order to control 
a MeArm robotic manipulator. When 4 consecutive predictions are the same,
the robotic manipulator moves.

@author: Ali Abdul Ameer Abbas
*/

#include <Servo.h> // Import the library for the Servo motors.
Servo baseMotor;   // Define the servo.
// Initialize counters.
int count_left = 0; 
int count_right = 0;
// Define the input Byte from Python.
String InputBytes;

// Set up the code.
void setup() {
  Serial.begin(9600);
  baseMotor.attach(9); // Define the pin of the servomotor.
  baseMotor.write(60); // Move the MeArm to the neutral position (60 degrees).
}

// Initialize the Loop.
void loop() {
  // Check if the Serial port is available.
  if (Serial.available()>0){
    // Read the input byte.
    InputBytes = Serial.readStringUntil('\n'); 
    Serial.flush(); 
    // If the Byte is "1" and there are 4 consecutive "1".
    // Move the MeArm to the Left.
    if (InputBytes == "1"){
      count_left += 1; // Sum the Left counter for counting the consecutives.
      count_right = 0; // Reset the Right counter.
      if (count_left == 4){
        baseMotor.write(120); // Move the MeArm base to 120 degrees.
        delay(100);
      }
    }
    // If the Byte is "2" and there are 4 consecutive "2".
    // Move the MeArm to the Left.
    if (InputBytes == "2"){
      count_left = 0;   // Sum the Right counter for counting the consecutives.
      count_right += 1; // Reset the Left counter.
      if (count_right == 4){
        baseMotor.write(0);  // Move the MeArm base to 0 degrees.
        delay(100); 
      }
    // If the Byte is "Reset", reset the counters.  
    if (InputBytes == "Reset"){
      count_left = 0;
      count_right= 0;
      }  
    }   
  } 
}
