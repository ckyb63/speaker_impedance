// This code is for the Arduino Nano 33 BLE Sense
// It reads the temperature, humidity, pressure, IMU data, and microphone data from the sensors
// and prints it to the Serial Monitor
// The code is based on the Arduino_HTS221.h and Arduino_LPS22HB.h libraries for the humidity and pressure sensors,
// the Arduino_LSM9DS1.h library for the IMU sensor (accelerometer, gyro, magnetometer),
// and the PDM library for the microphone

#include <Arduino_HTS221.h>   // For temperature and humidity
#include <Arduino_LPS22HB.h>  // For pressure sensor
#include <Arduino_LSM9DS1.h>  // For IMU (accelerometer, gyro, magnetometer)
#include <PDM.h>              // For the digital microphone

// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[256];

// Number of samples read
volatile int samplesRead;

// Variables for sound level calculation
float rawSoundLevel = 0;
float dBSPL = 0;      // Sound Pressure Level in dB
float dBA = 0;        // A-weighted sound level
float dBFS = 0;       // dB Full Scale

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
  // Find the peak-to-peak amplitude
  short maxSample = -32768;
  short minSample = 32767;
  
  for (int i = 0; i < numSamples; i++) {
    if (samples[i] > maxSample) {
      maxSample = samples[i];
    }
    if (samples[i] < minSample) {
      minSample = samples[i];
    }
  }
  
  // Calculate peak-to-peak amplitude
  float peakToPeak = maxSample - minSample;
  
  // Calculate RMS (Root Mean Square)
  float sum = 0;
  for (int i = 0; i < numSamples; i++) {
    sum += (float)samples[i] * (float)samples[i];
  }
  float rms = sqrt(sum / numSamples);
  
  // Store raw sound level (RMS)
  rawSoundLevel = rms;
  
  // Calculate dBFS (decibels relative to full scale)
  // Full scale is 32768 for 16-bit audio
  dBFS = 20.0 * log10(rms / 32768.0);
  
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
  
  // Calculate dB SPL (approximately dBA + 3 for typical environmental noise)
  dBSPL = dBA + 3.0;
  
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
    while (!Serial);

    // Initialize sensors
    if (!HTS.begin()) {
        Serial.println("Failed to initialize humidity and temperature sensor!");
    }
    if (!BARO.begin()) {
        Serial.println("Failed to initialize pressure sensor!");
    }
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU sensor!");
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
        while (1);  // Don't proceed if microphone initialization fails
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
    
    // Print calibration information
    Serial.println("Sound Level Calibration:");
    Serial.print("Point 1 (Low): RMS="); Serial.print(CAL_POINT1_RMS); 
    Serial.print(" -> "); Serial.print(CAL_POINT1_DBA); Serial.println(" dBA");
    Serial.print("Point 2 (Mid-Low): RMS="); Serial.print(CAL_POINT2_RMS); 
    Serial.print(" -> "); Serial.print(CAL_POINT2_DBA); Serial.println(" dBA");
    Serial.print("Point 3 (Mid-High): RMS="); Serial.print(CAL_POINT3_RMS); 
    Serial.print(" -> "); Serial.print(CAL_POINT3_DBA); Serial.println(" dBA");
    Serial.print("Point 4 (High): RMS="); Serial.print(CAL_POINT4_RMS); 
    Serial.print(" -> "); Serial.print(CAL_POINT4_DBA); Serial.println(" dBA");
    Serial.println("Using piecewise logarithmic calibration between points");
    
    // Print segment slopes and intercepts for verification
    Serial.println("Calibration Segments:");
    Serial.print("Segment 1: dBA = "); Serial.print(SLOPE1); Serial.print(" * log10(RMS) + "); Serial.println(INTERCEPT1);
    Serial.print("Segment 2: dBA = "); Serial.print(SLOPE2); Serial.print(" * log10(RMS) + "); Serial.println(INTERCEPT2);
    Serial.print("Segment 3: dBA = "); Serial.print(SLOPE3); Serial.print(" * log10(RMS) + "); Serial.println(INTERCEPT3);
    Serial.println();
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
    
    // Update other sensors at a lower frequency
    if (currentMillis - lastSensorUpdate >= SENSOR_UPDATE_INTERVAL) {
        lastSensorUpdate = currentMillis;
        
        // Read temperature and humidity
        float temperature = HTS.readTemperature() + TEMP_OFFSET;  // Apply temperature offset
        float humidity = HTS.readHumidity();

        // Read pressure
        float pressure = BARO.readPressure();

        // Read IMU data
        float ax, ay, az;
        float gx, gy, gz;
        float mx, my, mz;
        
        if (IMU.accelerationAvailable()) {
            IMU.readAcceleration(ax, ay, az);
        }
        if (IMU.gyroscopeAvailable()) {
            IMU.readGyroscope(gx, gy, gz);
        }
        if (IMU.magneticFieldAvailable()) {
            IMU.readMagneticField(mx, my, mz);
        }

        // Display all sensor readings
        Serial.println("----------------------------");
        Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" °C");
        Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
        Serial.print("Pressure: "); Serial.print(pressure); Serial.println(" hPa");
        
        Serial.print("Acceleration: X="); Serial.print(ax); Serial.print(" Y="); Serial.print(ay); Serial.print(" Z="); Serial.println(az);
        Serial.print("Gyroscope: X="); Serial.print(gx); Serial.print(" Y="); Serial.print(gy); Serial.print(" Z="); Serial.println(gz);
        Serial.print("Magnetometer: X="); Serial.print(mx); Serial.print(" Y="); Serial.print(my); Serial.print(" Z="); Serial.println(mz);
        
        // Display sound levels in different formats
        Serial.print("Raw Sound Level (RMS): "); Serial.println(rawSoundLevel);
        Serial.print("Sound Level (dBFS): "); Serial.println(dBFS);
        Serial.print("Sound Level (dB SPL): "); Serial.println(dBSPL);
        Serial.print("Sound Level (dBA): "); Serial.println(dBA);
        Serial.print("Sound Level (dBA avg): "); Serial.println(soundAverage);
        
        // Debug output for calibration verification
        if (rawSoundLevel >= 90 && rawSoundLevel <= 110) {
            Serial.print("CALIBRATION CHECK - RMS ~100: Expected ~65 dBA, Got: "); Serial.println(dBA);
        }
        
        Serial.println("----------------------------");
    }
}
