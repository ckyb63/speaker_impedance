// This code is for the Arduino Nano 33 BLE Sense
// It reads the temperature, humidity, pressure, and IMU data from the sensors and prints it to the Serial Monitor
// The code is based on the Arduino_HTS221.h and Arduino_LPS22HB.h libraries for the humidity and pressure sensors
// and the Arduino_LSM9DS1.h library for the IMU sensor (accelerometer, gyro, magnetometer)

#include <Arduino_HTS221.h>   // For temperature and humidity
#include <Arduino_LPS22HB.h>  // For pressure sensor
#include <Arduino_LSM9DS1.h>  // For IMU (accelerometer, gyro, magnetometer)

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
}

void loop() {
    // Read temperature and humidity
    float temperature = HTS.readTemperature();
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

    // Display readings
    Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" °C");
    Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
    Serial.print("Pressure: "); Serial.print(pressure); Serial.println(" hPa");
    
    Serial.print("Acceleration: X="); Serial.print(ax); Serial.print(" Y="); Serial.print(ay); Serial.print(" Z="); Serial.println(az);
    Serial.print("Gyroscope: X="); Serial.print(gx); Serial.print(" Y="); Serial.print(gy); Serial.print(" Z="); Serial.println(gz);
    Serial.print("Magnetometer: X="); Serial.print(mx); Serial.print(" Y="); Serial.print(my); Serial.print(" Z="); Serial.println(mz);
    
    Serial.println("----------------------------");
    delay(1000);  // Wait 1 second before next reading
}
