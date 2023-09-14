#include <SoftwareSerial.h>   //header file of software serial port
SoftwareSerial mySerial(13, 14); //define software serial port name as Serial1 and define pin2 as RX & pin3 as TX
 
int dist;                     //actual distance measurements of LiDAR
int strength;                 //signal strength of LiDAR
int check;                    //save check value
int i;
int uart[9];                   //save data measured by LiDAR
const int HEADER = 0x59;      //frame header of data package
 
 
void setup()
{
  Serial.begin(115200);         //set bit rate of serial port connecting Arduino with computer
  mySerial.begin(115200);      //set bit rate of serial port connecting LiDAR with Arduino
}
 
 
void loop() {
  if (mySerial.available())                //check if serial port has data input
  {
    if (mySerial.read() == HEADER)        //assess data package frame header 0x59
    {
      uart[0] = HEADER;
      if (mySerial.read() == HEADER)      //assess data package frame header 0x59
      {
        uart[1] = HEADER;
        for (i = 2; i < 9; i++)         //save data in array
        {
          uart[i] = mySerial.read();
        }
        check = uart[0] + uart[1] + uart[2] + uart[3] + uart[4] + uart[5] + uart[6] + uart[7];
        if (uart[8] == (check & 0xff))        //verify the received data as per protocol
        {
          dist = uart[2] + uart[3] * 256;     //calculate distance value
          strength = uart[4] + uart[5] * 256; //calculate signal strength value
          Serial.print("distance = ");
          Serial.print(dist);                 //output measure distance value of LiDAR
          Serial.print('\t');
          Serial.print("strength = ");
          Serial.print(strength);             //output signal strength value
          Serial.print('\n');
        }
      }
    }
  }
}
