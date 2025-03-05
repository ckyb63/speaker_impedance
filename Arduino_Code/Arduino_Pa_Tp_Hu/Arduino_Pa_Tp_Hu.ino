// This code is for the Arduino Uno with DHT22/AM2302 sensor
// It reads the temperature and humidity from the DHT22 sensor and prints it to the Serial Monitor
// Format is compatible with the Auto_Impedance_PyQt6.py GUI

// Author: Max Chen
// Version: 25.03.05-1
// Updated for Arduino Uno + DHT22 - Analog Discovery compatibility fix

#include <DHT.h>  // Include DHT library

#define DHTPIN 2       // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22  // DHT 22 (AM2302)

DHT dht(DHTPIN, DHTTYPE);  // Initialize DHT sensor

// Store last valid readings to use as fallbacks
float lastValidTemp = 25.0;    // Default room temperature
float lastValidHumidity = 50.0; // Default humidity

// Variables to manage timing and communication
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 1000; // Send data once per second
const unsigned long sensorInterval = 2000; // Read sensor every 2 seconds
unsigned long lastSensorTime = 0;

void setup() 
{
    Serial.begin(115200);  // Match baud rate with Python GUI (115200)
    
    // Empty the serial buffer
    while(Serial.available()) Serial.read();
    
    dht.begin();  // Initialize the DHT sensor
    
    // Small delay for sensor to stabilize
    delay(2000);
    
    // Send initial values
    sendSensorData();
}

void loop() {
    // Only read sensor at specific intervals
    unsigned long currentMillis = millis();
    
    // Process any incoming commands from the GUI
    processSerialCommands();
    
    // Read sensor at sensorInterval
    if (currentMillis - lastSensorTime >= sensorInterval) 
    {
        readSensor();
        lastSensorTime = currentMillis;
    }
    
    // Send data at sendInterval
    if (currentMillis - lastSendTime >= sendInterval) 
    {
        sendSensorData();
        lastSendTime = currentMillis;
    }
    
    // Small delay to prevent blocking the CPU
    delay(10);
}

void readSensor() {
    // Reading temperature and humidity
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();  // Read temperature in Celsius

    // Check if any reads failed and use last valid values
    if (!isnan(humidity)) lastValidHumidity = humidity;
    if (!isnan(temperature)) lastValidTemp = temperature;
}

void sendSensorData() {
    // Send readings in the comma-separated format expected by the Python GUI
    // Always provide values, even if using last valid readings
    Serial.print(lastValidTemp, 2); Serial.print(",");  // 2 decimal places for consistency
    Serial.print(lastValidHumidity, 2); Serial.print("\n");
}

void processSerialCommands() {
    // If any commands come in from the GUI, process them
    if (Serial.available() > 0) 
    {
        char command = Serial.read();
        
        // Simple command protocol can be added here if needed
        // For now, just clear the buffer
        while(Serial.available()) Serial.read();
    }
}
