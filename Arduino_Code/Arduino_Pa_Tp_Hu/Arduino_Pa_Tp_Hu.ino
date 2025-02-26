// This code is for the Arduino Nano 33 BLE Sense
// It reads the temperature, humidity, and pressure from the sensors and prints it to the Serial Monitor
// The code is based on the Arduino_HTS221.h and Arduino_LPS22HB.h libraries for the humidity and pressure sensors

// Author: Max Chen
// Date: 2025-02-26
// Version: 25.06

#include <Arduino_HTS221.h>   // For temperature and humidity
#include <Arduino_LPS22HB.h>  // For pressure sensor

void setup() {
    Serial.begin(115200);
    while (!Serial);

    // Initialize sensors
    if (!HTS.begin()) {
        Serial.println("Failed to initialize humidity and temperature sensor!");
    }
    if (!BARO.begin()) {
        Serial.println("Failed to initialize pressure sensor!");
    }
}

void loop() {
    // Read temperature and humidity
    float temperature = HTS.readTemperature();
    float humidity = HTS.readHumidity();

    // Read pressure
    float pressure = BARO.readPressure();

    // Display readings in a comma-separated format
    Serial.print(temperature); Serial.print(","); 
    Serial.print(humidity); Serial.print(","); 
    Serial.println(pressure);  // Ensure pressure is on a new line for clarity
    delay(500);
}
