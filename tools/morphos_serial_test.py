"""
PROJECT MORPHOS - PHASE 2.3
Serial Communication Test Script
Python → Arduino Traffic Controller
"""

import serial
import serial.tools.list_ports
import time
import sys

# SETTINGS
BAUD_RATE = 9600
TIMEOUT = 1

def list_available_ports():
    """Show all COM ports"""
    print("\nAvailable COM Ports:")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  No ports found!")
        return None
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")
    return ports

def connect_to_arduino(port_name=None):
    """Connect to Arduino"""
    if port_name is None:
        # Auto-detect or ask user
        ports = list_available_ports()
        if ports and len(ports) == 1:
            port_name = ports[0].device
            print(f"\nAuto-selected: {port_name}")
        else:
            port_name = input("\nEnter COM port (e.g., COM3 or COM4): ").strip()
    
    try:
        print(f"Connecting to {port_name} at {BAUD_RATE} baud...")
        ser = serial.Serial(port_name, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for Arduino reset after connection
        print(f"✓ Connected! Arduino is ready.")
        return ser
    except serial.SerialException as e:
        print(f"✗ Error: {e}")
        return None

def send_emergency(ser):
    """Send '1' - Force Emergency Green"""
    if ser and ser.is_open:
        ser.write(b'1')
        print(">>> SENT: EMERGENCY ('1') - Green incoming after 1s safety...")
        
        # Optional: read ACK if you enabled it in Arduino
        time.sleep(0.1)
        if ser.in_waiting:
            ack = ser.read()
            print(f"<<< ARDUINO: {ack.decode('utf-8', errors='ignore')} (Acknowledged)")

def clear_emergency(ser):
    """Send '0' - Return to Normal Cycle"""
    if ser and ser.is_open:
        ser.write(b'0')
        print(">>> SENT: RESET ('0') - Returning to normal cycle...")
        
        time.sleep(0.1)
        if ser.in_waiting:
            ack = ser.read()
            print(f"<<< ARDUINO: {ack.decode('utf-8', errors='ignore')} (Acknowledged)")

def main():
    print("=" * 50)
    print("MORPHOS SERIAL TEST")
    print("Commands: 1=Emergency | 0=Reset | q=Quit")
    print("=" * 50)
    
    # Connect
    ser = connect_to_arduino()
    if not ser:
        print("Failed to connect. Check USB cable and COM port.")
        sys.exit(1)
    
    print("\nTest Mode Active")
    print("Watch your traffic light!")
    
    try:
        while True:
            cmd = input("\nEnter command (1/0/q): ").strip().lower()
            
            if cmd == '1':
                send_emergency(ser)
            elif cmd == '0':
                clear_emergency(ser)
            elif cmd == 'q':
                print("Closing connection...")
                break
            else:
                print("Invalid command. Use 1, 0, or q")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("✓ Serial port closed")

if __name__ == "__main__":
    main()