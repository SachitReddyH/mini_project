#!/bin/bash

echo "Starting Indian Sign Language Translator..."
echo "The application will run in the background."
echo "To stop the application, press Ctrl+C."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the application in the background
nohup python run_app.py --host 127.0.0.1 --port 5000 > logs/console.log 2>&1 &

# Get the PID of the background process
APP_PID=$!

echo "Application started with PID: $APP_PID"
echo "Check logs/console.log for output."
echo "You can access the application at http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the application..."

# Function to handle Ctrl+C
function cleanup {
    echo "Stopping application..."
    kill $APP_PID
    echo "Application stopped."
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup SIGINT

# Keep the script running
while true; do
    sleep 1
done 