#!/usr/bin/env python3
# test_serial.py
#
# Simple script to test serial connection with physical loopback
# Connect TX to RX pins on the Raspberry Pi to create a loopback
# 
# Usage: python3 test_serial.py

import serial
import time

# Use same settings as main script
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 115200

def test_serial_loopback():
    print(f"Testing serial loopback on {SERIAL_PORT} at {BAUD_RATE} baud")
    print("Make sure TX and RX pins are physically connected (loopback)")
    
    try:
        # Open serial port
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1.0  # Longer timeout for testing
        )
        
        print(f"[OK] Serial port {SERIAL_PORT} opened successfully")
        
        # Flush any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send test messages
        for i in range(5):
            test_msg = f"TEST{i:03d}\n".encode('ascii')
            print(f"Sending: {test_msg.decode().strip()}")
            
            ser.write(test_msg)
            time.sleep(0.1)  # Give time for data to loop back
            
            # Read response (should be the same if loopback is working)
            if ser.in_waiting > 0:
                response = ser.readline()
                print(f"Received: {response.decode().strip()} [SUCCESS]")
            else:
                print("No data received - loopback failed")
        
        # Simple continuous echo test
        print("\nStarting continuous echo test (press Ctrl+C to exit)...")
        print("Type a message and see if it comes back:")
        
        ser.reset_input_buffer()
        
        while True:
            # Send position command like main program
            x = int(time.time() * 10) % 2857  # Value changes over time
            y = 1000
            cmd = f"M{x:04d}{y:04d}\n"
            
            print(f"Sending: {cmd.strip()}")
            ser.write(cmd.encode('ascii'))
            
            # Read response
            time.sleep(0.1)
            if ser.in_waiting > 0:
                response = ser.readline()
                print(f"Received: {response.decode().strip()}")
            else:
                print("No response received")
            
            time.sleep(1.0)  # Wait 1 second between tests
            
    except KeyboardInterrupt:
        print("\nTest ended by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    test_serial_loopback() 