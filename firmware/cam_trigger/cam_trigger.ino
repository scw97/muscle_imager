/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO 
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN takes care 
  of use the correct LED pin whatever is the board used.
  If you want to know what pin the on-board LED is connected to on your Arduino model, check
  the Technical Specs of your board  at https://www.arduino.cc/en/Main/Products
  
  This example code is in the public domain.

  modified 8 May 2014
  by Scott Fitzgerald
  
  modified 2 Sep 2016
  by Arturo Guadalupi
*/
int CAM1_LED_PIN = 46;
int CAM2_LED_PIN = 48;
int SYNC_LED_PIN = 50;
int framecount = 0;
bool framestate = false;
int pulsestate = 1;

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(CAM1_LED_PIN, OUTPUT);
  pinMode(CAM2_LED_PIN, OUTPUT);
  pinMode(SYNC_LED_PIN, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
  if (framecount > 100){
    framestate = !framestate;
    framecount = 0;
  }
    framecount +=1;
    digitalWrite(CAM1_LED_PIN, HIGH);   // turn the LED on (HIGH is the voltage level)
    digitalWrite(CAM2_LED_PIN, HIGH);
    delay(15);                      
    digitalWrite(CAM2_LED_PIN, LOW);    // turn the LED off by making the voltage LOW
    delayMicroseconds(100);
    digitalWrite(CAM2_LED_PIN, HIGH);
    delay(15);
    digitalWrite(CAM1_LED_PIN, LOW);
    digitalWrite(CAM2_LED_PIN, LOW);
    delayMicroseconds(100);
    digitalWrite(SYNC_LED_PIN, framestate);
    
}

