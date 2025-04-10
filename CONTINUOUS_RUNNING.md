# Running the Indian Sign Language Translator Continuously

This document provides instructions on how to run the Indian Sign Language Translator application continuously on different operating systems.

## Windows

### Option 1: Using the Batch Script

1. Double-click on `start_app.bat` to start the application in the background.
2. The application will run continuously until you close the command prompt window or press any key.
3. You can access the application at http://127.0.0.1:5000

### Option 2: Using Task Scheduler

1. Open Task Scheduler (search for it in the Start menu)
2. Click on "Create Basic Task"
3. Enter a name (e.g., "Indian Sign Language Translator") and description
4. Set the trigger to "When the computer starts"
5. Select "Start a program" as the action
6. Browse to your Python executable (e.g., `C:\Python39\python.exe`)
7. Add the arguments: `run_app.py --host 127.0.0.1 --port 5000`
8. Set the "Start in" field to your project directory
9. Complete the wizard and check "Open the Properties dialog" before finishing
10. In the Properties dialog, go to the "Settings" tab and uncheck "Stop the task if it runs longer than:"
11. Click OK to save

## Linux/macOS

### Option 1: Using the Shell Script

1. Open a terminal and navigate to your project directory
2. Make the script executable: `chmod +x start_app.sh`
3. Run the script: `./start_app.sh`
4. The application will run continuously until you press Ctrl+C
5. You can access the application at http://127.0.0.1:5000

### Option 2: Using systemd (Linux only)

1. Edit the `indian-sign-language.service` file to set your username and correct paths
2. Copy the service file to the systemd directory:
   ```
   sudo cp indian-sign-language.service /etc/systemd/system/
   ```
3. Reload systemd:
   ```
   sudo systemctl daemon-reload
   ```
4. Enable and start the service:
   ```
   sudo systemctl enable indian-sign-language.service
   sudo systemctl start indian-sign-language.service
   ```
5. Check the status:
   ```
   sudo systemctl status indian-sign-language.service
   ```

## Using Docker (All Platforms)

1. Build the Docker image:
   ```
   docker build -t indian-sign-language .
   ```
2. Run the container:
   ```
   docker run -d -p 5000:5000 --name indian-sign-language indian-sign-language
   ```
3. The application will run continuously in the container
4. You can access the application at http://localhost:5000

## Monitoring and Logs

- Logs are stored in the `logs` directory
- Check `logs/app.log` for application logs
- Check `logs/console.log` for console output

## Troubleshooting

If the application stops unexpectedly:

1. Check the logs for error messages
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Try running the application directly: `python run_app.py`
4. Check if the port is already in use by another application 