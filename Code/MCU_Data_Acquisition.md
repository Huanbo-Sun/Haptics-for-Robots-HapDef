# MCU_Data_Acquisition in [Arduino IDE](https://www.arduino.cc/en/Main/Software)
```
int val1 = 0;
int val2 = 0;
int val3 = 0;
int val4 = 0;
int val5 = 0;
int val6 = 0;
int val7 = 0;
int val8 = 0;
int val9 = 0;
int val10 =0;

void setup()
{
  Serial.begin(115200);              //  setup serial
}

void loop()
{
  analogReadResolution(12);
  val1 = analogRead(0);
  val2 = analogRead(1);
  val3 = analogRead(2);
  val4 = analogRead(3);
  val5 = analogRead(5);
  val6 = analogRead(6);
  val7 = analogRead(7);
  val8 = analogRead(8);
  val9 = analogRead(9);
  val10 = analogRead(10);
  Serial.print(val1);
  Serial.print(",");
  Serial.print(val2);
  Serial.print(",");
  Serial.print(val3);
  Serial.print(",");
  Serial.print(val4);
  Serial.print(",");
  Serial.print(val5);
  Serial.print(",");
  Serial.print(val6);
  Serial.print(",");
  Serial.print(val7);
  Serial.print(",");
  Serial.print(val8);
  Serial.print(",");
  Serial.print(val9);
  Serial.print(",");
  Serial.println(val10);

  delay(100);
    }
```
