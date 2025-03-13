// Arduino Nano 33 BLE Sense - Temperature, Humidity, and Sound Level Monitor
// This code reads temperature, humidity, and sound level data from the Nano 33 BLE Sense
// and outputs it in a comma-separated format compatible with the PyQt6 GUI.
// Format: temperature,humidity,rawSoundLevel,dBA

#include <Arduino_HTS221.h>   // For temperature and humidity
#include <PDM.h>              // For the digital microphone

// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[256];

// Number of samples read
volatile int samplesRead;

// Variables for sound level calculation
float rawSoundLevel = 0;
float dBA = 0;        // A-weighted sound level

// Calibration points from sound calibrator
// Point 1: RMS value of ~50 corresponds to 39 dBA (low level)
// Point 2: RMS value of ~100 corresponds to ~65 dBA (estimated intermediate point)
// Point 3: RMS value of ~230 corresponds to 94 dBA (medium level)
// Point 4: RMS value of ~2770 corresponds to 114 dBA (high level)
const float CAL_POINT1_RMS = 50.0;
const float CAL_POINT1_DBA = 39.0;
const float CAL_POINT2_RMS = 100.0;
const float CAL_POINT2_DBA = 65.0;  // Estimated value for better curve fitting
const float CAL_POINT3_RMS = 230.0;
const float CAL_POINT3_DBA = 94.0;
const float CAL_POINT4_RMS = 2770.0;
const float CAL_POINT4_DBA = 114.0;

// Pre-calculated slopes and intercepts for each segment
// For segment 1 (between points 1 and 2)
const float SLOPE1 = (CAL_POINT2_DBA - CAL_POINT1_DBA) / (log10(CAL_POINT2_RMS) - log10(CAL_POINT1_RMS));
const float INTERCEPT1 = CAL_POINT1_DBA - SLOPE1 * log10(CAL_POINT1_RMS);

// For segment 2 (between points 2 and 3)
const float SLOPE2 = (CAL_POINT3_DBA - CAL_POINT2_DBA) / (log10(CAL_POINT3_RMS) - log10(CAL_POINT2_RMS));
const float INTERCEPT2 = CAL_POINT2_DBA - SLOPE2 * log10(CAL_POINT2_RMS);

// For segment 3 (between points 3 and 4)
const float SLOPE3 = (CAL_POINT4_DBA - CAL_POINT3_DBA) / (log10(CAL_POINT4_RMS) - log10(CAL_POINT3_RMS));
const float INTERCEPT3 = CAL_POINT3_DBA - SLOPE3 * log10(CAL_POINT3_RMS);

const float TEMP_OFFSET = -2.0;  // Temperature offset correction (-2°C)

// Timing variables
unsigned long lastSensorUpdate = 0;
const unsigned long SENSOR_UPDATE_INTERVAL = 1000;  // Update sensors every 1000ms

// Running average for sound levels
const int SOUND_SAMPLES = 5;
float soundLevels[SOUND_SAMPLES];
int soundSampleIndex = 0;
float soundAverage = 0;

// Callback function for PDM data ready
void onPDMdata() {
  // Query the number of bytes available
  int bytesAvailable = PDM.available();

  // Read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;
}

// Function to calculate sound level from raw samples
void calculateSoundLevels(short* samples, int numSamples) {
  // Calculate RMS (Root Mean Square)
  float sum = 0;
  for (int i = 0; i < numSamples; i++) {
    sum += (float)samples[i] * (float)samples[i];
  }
  float rms = sqrt(sum / numSamples);
  
  // Store raw sound level (RMS)
  rawSoundLevel = rms;
  
  // Calculate calibrated dBA using piecewise logarithmic mapping
  if (rms <= 1.0) {
    dBA = 0.0;  // Below threshold of measurement
  } 
  else if (rms < CAL_POINT1_RMS) {
    // For very low levels, use a linear approximation
    dBA = (CAL_POINT1_DBA / CAL_POINT1_RMS) * rms;
  }
  else if (rms < CAL_POINT2_RMS) {
    // Between point 1 and point 2
    dBA = SLOPE1 * log10(rms) + INTERCEPT1;
  }
  else if (rms < CAL_POINT3_RMS) {
    // Between point 2 and point 3
    dBA = SLOPE2 * log10(rms) + INTERCEPT2;
  }
  else {
    // Between point 3 and point 4 (or above point 4)
    dBA = SLOPE3 * log10(rms) + INTERCEPT3;
  }
  
  // Update running average
  soundLevels[soundSampleIndex] = dBA;
  soundSampleIndex = (soundSampleIndex + 1) % SOUND_SAMPLES;
  
  // Calculate average
  sum = 0;
  for (int i = 0; i < SOUND_SAMPLES; i++) {
    sum += soundLevels[i];
  }
  soundAverage = sum / SOUND_SAMPLES;
}

void setup() {
  Serial.begin(115200);
  
  // Initialize sensors
  if (!HTS.begin()) {
    Serial.println("Failed to initialize humidity and temperature sensor!");
    while (1); // Don't proceed if sensor initialization fails
  }
  
  // Configure the PDM microphone
  PDM.onReceive(onPDMdata);
  
  // Optionally set the gain, defaults to 20
  PDM.setGain(30);
  
  // Initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to initialize PDM microphone!");
    while (1); // Don't proceed if microphone initialization fails
  }
  
  // Initialize built-in LEDs for sound level indication
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  
  // Turn off LEDs (they are active LOW)
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  
  // Initialize sound level array
  for (int i = 0; i < SOUND_SAMPLES; i++) {
    soundLevels[i] = 0;
  }
  
  Serial.println("Arduino Nano 33 BLE Sense - Temperature, Humidity, and Sound Level Monitor");
  Serial.println("Data format: temperature,humidity,rawSoundLevel,dBA");
  delay(1000);
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Process microphone data if available
  if (samplesRead) {
    // Calculate sound levels
    calculateSoundLevels(sampleBuffer, samplesRead);
    
    // Visualize sound level with LEDs
    // Using dBA for LED indication (more relevant to human hearing)
    if (dBA >= 85) {
      // High sound level - Red LED (potentially harmful)
      digitalWrite(LEDR, LOW);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, HIGH);
    } else if (dBA >= 65 && dBA < 85) {
      // Medium sound level - Blue LED (moderate)
      digitalWrite(LEDB, LOW);
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
    } else {
      // Low sound level - Green LED (safe)
      digitalWrite(LEDG, LOW);
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDB, HIGH);
    }
    
    // Clear the read count
    samplesRead = 0;
  }
  
  // Update and send sensor data at a regular interval
  if (currentMillis - lastSensorUpdate >= SENSOR_UPDATE_INTERVAL) {
    lastSensorUpdate = currentMillis;
    
    // Read temperature and humidity
    float temperature = HTS.readTemperature() + TEMP_OFFSET;  // Apply temperature offset
    float humidity = HTS.readHumidity();
    
    // Output data in the format expected by the PyQt6 GUI
    // Format: temperature,humidity,rawSoundLevel,dBA
    Serial.print(temperature);
    Serial.print(",");
    Serial.print(humidity);
    Serial.print(",");
    Serial.print(rawSoundLevel);
    Serial.print(",");
    Serial.println(dBA);
    
    // Also print a more detailed output for debugging (commented out for production)
    /*
    Serial.println("----------------------------");
    Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" °C");
    Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
    Serial.print("Raw Sound Level (RMS): "); Serial.println(rawSoundLevel);
    Serial.print("Sound Level (dBA): "); Serial.println(dBA);
    Serial.print("Sound Level (dBA avg): "); Serial.println(soundAverage);
    Serial.println("----------------------------");
    */
  }
}
