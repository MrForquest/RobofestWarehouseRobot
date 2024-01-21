#include <Wire.h>
#define SLAVE_ADDRESS 0x04

void setup() {
  // Serial.begin(9600);
  Wire.begin(SLAVE_ADDRESS);
  Wire.onReceive(receiveData);
  Wire.onRequest(sendData);
  Serial.begin(19200);
  Serial.println("Ready!");

}


int flag = 0;
byte val, resSerial;
void loop() {
  if (flag == 1)
  {
    Serial.write(val);
    val = 0;
    while (Serial.available()) {
      resSerial = Serial.read();
      // Serial.println(resSerial);
    }
    flag = 0;
  }
}
void receiveData(int byteCount)
{
  while (Wire.available() > 0)
  {
    val = Wire.read();
    flag = 1;
  }
}
// Функция для отправки данных.
void sendData()
{
  Wire.write(resSerial);
  resSerial = 0;
  flag = 0;
}
