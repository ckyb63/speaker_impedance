// This code is for the Arduino Uno with DHT22/AM2302 sensor
// It reads the temperature and humidity from the DHT22 sensor and prints it to the Serial Monitor
// Format is compatible with the Auto_Impedance_PyQt6.py GUI

// Author: Max Chen
// Version: 25.03.05-1
// Updated for Arduino Uno + DHT22 - Analog Discovery compatibility fix

#include <DHT.h>          // Include DHT library
#include <ArduinoSound.h>  // Include ArduinoSound library

#define DHTPIN 2       // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22  // DHT 22 (AM2302)

// Audio processing parameters
#define SAMPLES_TO_READ 256  // Reduced buffer size for better real-time response
const int sampleRate = 16000;  // Sample rate in Hz

DHT dht(DHTPIN, DHTTYPE);  // Initialize DHT sensor
AudioInI2S audioInput;     // Create audio input instance

// Store last valid readings to use as fallbacks
float lastValidTemp = 25.0;    // Default room temperature
float lastValidHumidity = 50.0; // Default humidity
float lastValidDb = 0.0;       // Default dB level

// Variables to manage timing and communication
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 1000; // Send data once per second
const unsigned long sensorInterval = 2000; // Read sensor every 2 seconds
unsigned long lastSensorTime = 0;

// Buffer for audio samples
short samples[SAMPLES_TO_READ];

void setup() 
{
    Serial.begin(115200);  // Match baud rate with Python GUI (115200)
    
    // Empty the serial buffer
    while(Serial.available()) Serial.read();
    
    dht.begin();  // Initialize the DHT sensor
    
    // Initialize audio input
    if (!audioInput.begin()) {
        Serial.println("Failed to initialize audio input!");
        while (1); // Don't proceed if initialization failed
    }
    
    // Configure audio input
    audioInput.setGain(5);  // Adjust gain as needed (0-255)
    
    // Small delay for sensors to stabilize
    delay(2000);
    
    // Send initial values
    sendSensorData();
}

// Function to calculate dB level from raw samples
float calculateDB(short* samples, int numSamples) {
    float sum = 0.0;
    
    // Calculate RMS
    for(int i = 0; i < numSamples; i++) {
        // Normalize to -1.0 to 1.0 (16-bit samples)
        float sample = samples[i] / 32768.0;
        sum += sample * sample;
    }
    
    float rms = sqrt(sum / numSamples);
    
    // Convert to dB (using -60dB as reference for silence)
    float db = 20.0 * log10(rms) + 60.0;
    
    // Clamp values
    if(db < 0.0) db = 0.0;
    if(db > 120.0) db = 120.0;
    
    return db;
}

void readMicrophone() {
    // Check if samples are available
    if (audioInput.available()) {
        // Read audio samples
        audioInput.read(samples, SAMPLES_TO_READ);
        
        float db = calculateDB(samples, SAMPLES_TO_READ);
        if(!isnan(db)) lastValidDb = db;
    }
}

void readSensor() {
    // Reading temperature and humidity
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();  // Read temperature in Celsius

    // Check if any reads failed and use last valid values
    if (!isnan(humidity)) lastValidHumidity = humidity;
    if (!isnan(temperature)) lastValidTemp = temperature;
    
    // Read microphone data
    readMicrophone();
}

void sendSensorData() {
    // Send readings in the comma-separated format expected by the Python GUI
    // Now includes dB level
    Serial.print(lastValidTemp, 2); Serial.print(",");
    Serial.print(lastValidHumidity, 2); Serial.print(",");
    Serial.print(lastValidDb, 2); Serial.print("\n");
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
