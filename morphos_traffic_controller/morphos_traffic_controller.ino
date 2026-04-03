/*
 * PROJECT MORPHOS - ENHANCED ARDUINO TRAFFIC CONTROLLER
 * Original pinout preserved + improved serial handling + safety features
 * 
 * SAFETY FEATURES:
 * - Watchdog timer: Auto-reset to normal if Python crashes in Emergency Mode
 */

// Pin definitions (ORIGINAL PINS PRESERVED)
const int RED_PIN = 3;
const int YELLOW_PIN = 2;
const int GREEN_PIN = 5;

// Timing constants (ORIGINAL)
const unsigned long RED_DURATION = 3000;
const unsigned long YELLOW_DURATION = 2000;
const unsigned long GREEN_DURATION = 3000;
const unsigned long ALL_RED_DURATION = 1000;

// WATCHDOG SAFETY TIMER - Auto-reset if Python crashes
const unsigned long WATCHDOG_TIMEOUT = 5000;  // 5 seconds

// States (ORIGINAL + improved)
enum States { RED, YELLOW, GREEN, ALL_RED_CLEAR, EMERGENCY_GREEN };

States currentState = RED;
unsigned long stateStartTime;
bool emergencyActive = false;
bool headingToGreen = true;

// Watchdog timer variable - tracks last time we received a command
unsigned long lastCommandTime = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(100);
  
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  
  // Initial state: RED (safer than GREEN)
  setState(RED);
  
  // Initialize watchdog timer
  lastCommandTime = millis();
  
  delay(2000);
  Serial.println("MORPHOS READY");  // Boot confirmation
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Enhanced serial handling with ACK
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == '1') {
      if (!emergencyActive) {
        emergencyActive = true;
        Serial.print("ACK:1");  // Python expects this
        Serial.flush();
      }
      // Update watchdog timer on any valid command
      lastCommandTime = currentMillis;
    } 
    else if (cmd == '0') {
      if (emergencyActive) {
        emergencyActive = false;
        Serial.print("ACK:0");
        Serial.flush();
      }
      // Update watchdog timer on any valid command
      lastCommandTime = currentMillis;
    }
    
    // Clear buffer
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
  
  // WATCHDOG SAFETY CHECK
  // If emergency is active but we haven't heard from Python in 5 seconds,
  // assume Python crashed and auto-reset to normal mode
  if (emergencyActive && (currentMillis - lastCommandTime > WATCHDOG_TIMEOUT)) {
    emergencyActive = false;
    Serial.print("WATCHDOG:RESET");  // Notify that watchdog triggered
    Serial.flush();
    // Force immediate return to RED state for safety
    if (currentState == EMERGENCY_GREEN) {
      setState(RED);
    }
  }
  
  // Emergency override logic (ORIGINAL + safety)
  if (emergencyActive) {
    if (currentState != EMERGENCY_GREEN && currentState != ALL_RED_CLEAR) {
      setState(ALL_RED_CLEAR);  // Safe transition
    }
  } else if (currentState == EMERGENCY_GREEN) {
    setState(RED);  // Safe return to normal
  }
  
    // State machine (Fixed Normal Traffic Cycle)
  unsigned long elapsed = currentMillis - stateStartTime;
  
  switch (currentState) {
    case GREEN:
      if (elapsed >= GREEN_DURATION) {
        setState(YELLOW);
      }
      break;
      
    case YELLOW:
      if (elapsed >= YELLOW_DURATION) {
        if (emergencyActive) {
          setState(ALL_RED_CLEAR); // Safety buffer before emergency green
        } else {
          setState(RED);
        }
      }
      break;
      
    case RED:
      if (elapsed >= RED_DURATION) {
        if (emergencyActive) {
          setState(EMERGENCY_GREEN);
        } else {
          setState(GREEN);
        }
      }
      break;
      
    case ALL_RED_CLEAR:
      if (elapsed >= ALL_RED_DURATION) {
        setState(EMERGENCY_GREEN);
      }
      break;
      
    case EMERGENCY_GREEN:
      // Stay in emergency green indefinitely (unless watchdog triggers or Python clears it)
      break;
  }
}

void setState(States newState) {
  if (currentState != newState) {
    currentState = newState;
    stateStartTime = millis();
    updateLEDs();
  }
}

void updateLEDs() {
  digitalWrite(RED_PIN, (currentState == RED || currentState == ALL_RED_CLEAR) ? HIGH : LOW);
  digitalWrite(YELLOW_PIN, (currentState == YELLOW) ? HIGH : LOW);
  digitalWrite(GREEN_PIN, (currentState == GREEN || currentState == EMERGENCY_GREEN) ? HIGH : LOW);
}