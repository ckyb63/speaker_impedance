// Arduino Nano 33 BLE Sense - Temperature, Humidity, and Sound Level Monitor
// This code reads temperature, humidity, and sound level data from the Nano 33 BLE Sense
// and outputs it in a comma-separated format compatible with the PyQt6 GUI.
// Format: temperature,humidity,rawSoundLevel,dBA

#include <Arduino_HTS221.h>   // For temperature and humidity
#include <Arduino_LPS22HB.h> // For pressure
#include <PDM.h>              // For the digital microphone

// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[512];  // Increased buffer size for better averaging

// Number of samples read
volatile int samplesRead;

// Variables for sound level calculation
float rawSoundLevel = 0;
float dBA = 0;        // A-weighted sound level
float pressure = 0;   // Pressure in hPa

// Multi-segment logarithmic curve calibration parameters
// Four-point calibration based on measured values
// Point 1: RMS 46 corresponds to 36 dBA (quiet room)
// Point 2: RMS 69 corresponds to 53.2 dBA (low-medium level)
// Point 3: RMS 171 corresponds to 79 dBA (medium level)
// Point 4: RMS 500 corresponds to 95 dBA (high level)
const float CAL_POINT1_RMS = 45.0;
const float CAL_POINT1_DBA = 35.0;
const float CAL_POINT2_RMS = 60.0;
const float CAL_POINT2_DBA = 50.0;
const float CAL_POINT3_RMS = 190.0;  
const float CAL_POINT3_DBA = 70.0;
const float CAL_POINT4_RMS = 1000.0;  // New higher calibration point  
const float CAL_POINT4_DBA = 100.0;   // New higher calibration point

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

// No room offset needed since we're calibrating directly to the reference sound meter
const float SILENT_ROOM_OFFSET = 0.0;

// Sensor offset corrections
const float TEMP_OFFSET = -4.0;  // Temperature offset correction (-2.5°C)
const float HUMIDITY_OFFSET = 3.5;  // Humidity offset correction (-5%) - adjust as needed

// Timing variables
unsigned long lastSensorUpdate = 0;
const unsigned long SENSOR_UPDATE_INTERVAL = 1000;  // Update sensors every 1000ms

// Improved running average for sound levels - increased sample count
const int SOUND_SAMPLES = 35;  // Increased from 20 to 30 for more stability
float soundLevels[SOUND_SAMPLES];
int soundSampleIndex = 0;
float soundAverage = 0;

// Exponential smoothing factor (alpha)
// Higher values (closer to 1) give more weight to recent readings
// Lower values (closer to 0) give more weight to past readings
const float ALPHA = 0.1;  // Reduced from 0.15 to 0.1 for smoother transitions

// Values for median filtering to remove outliers
const int MEDIAN_SAMPLES = 9;
float recentValues[MEDIAN_SAMPLES];
int medianIndex = 0;

float smoothedDBA = 0;

// Variables for LED debouncing
unsigned long lastLEDChange = 0;
const unsigned long LED_DEBOUNCE_TIME = 500;  // Minimum time between LED state changes (ms)
int currentLEDState = 0;  // 0 = green, 1 = blue, 2 = red

// Sample accumulation
const int MIN_SAMPLES_BEFORE_CALCULATION = 5;  // Increased from 3 to 5
int sampleBatchCount = 0;
float accumulatedRMS = 0;

// Function to sort array for median calculation
void bubbleSort(float arr[], int size) {
  for (int i = 0; i < size - 1; i++) {
    for (int j = 0; j < size - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        // Swap
        float temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}

// Function to get median value from array
float getMedian(float arr[], int size) {
  // Create a copy to avoid modifying the original array
  float temp[size];
  for (int i = 0; i < size; i++) {
    temp[i] = arr[i];
  }
  
  // Sort the copy
  bubbleSort(temp, size);
  
  // Return median
  if (size % 2 == 0) {
    return (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
  } else {
    return temp[size / 2];
  }
}

// Callback function for PDM data ready
void onPDMdata() {
  // Query the number of bytes available
  int bytesAvailable = PDM.available();

  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample
  samplesRead = bytesAvailable / 2;
}

// Function to calculate sound level from raw samples
float calculateRMS(short* samples, int numSamples) {
  // Calculate RMS (Root Mean Square)
  float sum = 0;
  for (int i = 0; i < numSamples; i++) {
    // Apply a moderate noise floor to filter out microphone self-noise
    float sample = abs((float)samples[i]);
    if (sample < 25) {  // Decreased from 35 to 20 - to capture more ambient noise
      sample = 0;
    }
    sum += sample * sample;
  }
  return sqrt(sum / numSamples);
}

// Function to convert RMS to dBA using the multi-segment logarithmic curve
float convertToDBA(float rms) {
  float result = 0;
  
  if (rms <= 8.0) {  // Reduced from 10.0 to 5.0
    return 0.0;  // Below threshold of measurement
  } 
  else if (rms < CAL_POINT1_RMS) {
    // For very low levels, use a linear approximation
    result = (CAL_POINT1_DBA / CAL_POINT1_RMS) * rms;
  }
  else if (rms < CAL_POINT2_RMS) {
    // Between point 1 and point 2 (very low to low)
    result = SLOPE1 * log10(rms) + INTERCEPT1;
  }
  else if (rms < CAL_POINT3_RMS) {
    // Between point 2 and point 3 (low to medium)
    result = SLOPE2 * log10(rms) + INTERCEPT2;
  }
  else if (rms < CAL_POINT4_RMS) {
    // Between point 3 and point 4 (medium to high)
    result = SLOPE3 * log10(rms) + INTERCEPT3;
  }
  else {
    // Above point 4 (very high) - extend the last segment
    result = SLOPE3 * log10(rms) + INTERCEPT3;
  }
  
  return result;
}

// Function to update the LED based on sound level with debouncing
void updateLED(float dBA) {
  unsigned long currentMillis = millis();
  int newLEDState;
  
  // Determine the appropriate LED state - updated thresholds to match professional scale
  if (dBA >= 85) {
    newLEDState = 2;  // Red - high level
  } else if (dBA >= 65) {
    newLEDState = 1;  // Blue - medium level
  } else {
    newLEDState = 0;  // Green - low level
  }
  
  // Only change LED if state is different and debounce time has passed
  if (newLEDState != currentLEDState && 
      (currentMillis - lastLEDChange > LED_DEBOUNCE_TIME)) {
    
    // Turn off all LEDs first (they are active LOW)
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    
    // Turn on the appropriate LED
    switch (newLEDState) {
      case 0:  // Green - low sound level
        digitalWrite(LEDG, LOW);
        break;
      case 1:  // Blue - medium sound level
        digitalWrite(LEDB, LOW);
        break;
      case 2:  // Red - high sound level
        digitalWrite(LEDR, LOW);
        break;
    }
    
    currentLEDState = newLEDState;
    lastLEDChange = currentMillis;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);  // Wait for serial or timeout after 5 seconds
  
  // Initialize sensors
  if (!HTS.begin()) {
    Serial.println("Failed to initialize humidity and temperature sensor!");
    while (1); // Don't proceed if sensor initialization fails
  }

  if (!BARO.begin()) {
    Serial.println("Failed to initialize pressure sensor!");
    while (1); // Don't proceed if sensor initialization fails
  }
  
  // Configure the PDM microphone
  PDM.onReceive(onPDMdata);
  
  // Increased gain for better match with reference sound meter
  PDM.setGain(30);  // Increased from 20 to 25 for better sensitivity
  
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
  
  // Initialize median filter array
  for (int i = 0; i < MEDIAN_SAMPLES; i++) {
    recentValues[i] = 0;
  }
  
  Serial.println("Arduino Nano 33 BLE Sense - Temperature, Humidity, and Sound Level Monitor");
  Serial.println("Data format: temperature,humidity,rawSoundLevel,dBA,smoothedDBA");
  delay(1000);
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Process microphone data if available
  if (samplesRead) {
    // Calculate RMS for this batch of samples
    float batchRMS = calculateRMS(sampleBuffer, samplesRead);
    
    // Accumulate RMS values
    accumulatedRMS += batchRMS;
    sampleBatchCount++;
    
    // Only proceed with calculations after collecting enough samples
    if (sampleBatchCount >= MIN_SAMPLES_BEFORE_CALCULATION) {
      // Calculate average RMS from accumulated samples
      rawSoundLevel = accumulatedRMS / sampleBatchCount;
      
      // Convert RMS to dBA using multi-segment logarithmic curve
      dBA = convertToDBA(rawSoundLevel);
      
      // Add to median filter
      recentValues[medianIndex] = dBA;
      medianIndex = (medianIndex + 1) % MEDIAN_SAMPLES;
      
      // Get median value to filter out spikes
      float medianDBA = getMedian(recentValues, MEDIAN_SAMPLES);
      
      // Apply exponential smoothing to the median-filtered value
      if (smoothedDBA == 0) {
        smoothedDBA = medianDBA;  // Initialize on first run
      } else {
        smoothedDBA = ALPHA * medianDBA + (1 - ALPHA) * smoothedDBA;
      }
      
      // Update running average
      soundLevels[soundSampleIndex] = smoothedDBA;
      soundSampleIndex = (soundSampleIndex + 1) % SOUND_SAMPLES;
      
      // Calculate average
      float sum = 0;
      for (int i = 0; i < SOUND_SAMPLES; i++) {
        sum += soundLevels[i];
      }
      soundAverage = sum / SOUND_SAMPLES;
      
      // Update LED indicator based on smoothed value
      updateLED(smoothedDBA);
      
      // Reset accumulation
      accumulatedRMS = 0;
      sampleBatchCount = 0;
    }
    
    // Clear the read count
    samplesRead = 0;
  }
  
  // Update and send sensor data at a regular interval
  if (currentMillis - lastSensorUpdate >= SENSOR_UPDATE_INTERVAL) {
    lastSensorUpdate = currentMillis;
    
    // Read temperature and humidity with offset corrections
    float temperature = HTS.readTemperature() + TEMP_OFFSET;
    float humidity = HTS.readHumidity() + HUMIDITY_OFFSET;
    
    // Read pressure
    pressure = BARO.readPressure();
    
    // Ensure humidity stays within valid range (0-100%)
    humidity = constrain(humidity, 0.0, 100.0);
    
    // Output data in the format expected by the PyQt6 GUI
    // Format: temperature,humidity,pressure,rawSoundLevel,smoothedDBA
    Serial.print(temperature);
    Serial.print(",");
    Serial.print(humidity);
    Serial.print(",");
    Serial.print(pressure);
    Serial.print(",");
    Serial.print(rawSoundLevel);
    Serial.print(",");
    Serial.println(soundAverage);
    
    // Debugging and Calibration use
    // Serial.println("----------------------------");
    // Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" °C");
    // Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
    // Serial.print("Pressure: "); Serial.print(pressure); Serial.println(" hPa");
    // Serial.print("Raw Sound Level (RMS): "); Serial.println(rawSoundLevel);
    // Serial.print("Sound Level (dBA): "); Serial.println(dBA);
    // Serial.print("Smoothed Sound Level (dBA): "); Serial.println(smoothedDBA);
    // Serial.print("Sound Level (dBA avg): "); Serial.println(soundAverage);
    // Serial.println("----------------------------");
  }
}