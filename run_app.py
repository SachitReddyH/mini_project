#!/usr/bin/env python
"""
Indian Sign Language Translator - Main Application Runner

This script runs the Flask web application with the Indian Sign Language Translator models.
It includes error handling and automatic restart capabilities to ensure continuous operation.
"""

import os
import sys
import argparse
import time
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_app(host, port, debug):
    """Run the Flask application."""
    try:
        # Import the Flask app
        from src.web.simple_app import app
        
        # Run the app
        logger.info(f"Starting Indian Sign Language Translator on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.error(f"Error running the application: {e}", exc_info=True)
        return False
    return True

def main():
    """Main function to run the application continuously."""
    # Set up logging
    global logger
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Indian Sign Language Translator')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--max-restarts', type=int, default=5, help='Maximum number of restart attempts')
    parser.add_argument('--restart-delay', type=int, default=5, help='Delay in seconds between restart attempts')
    args = parser.parse_args()
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Run the application continuously
    restart_count = 0
    while True:
        logger.info(f"Starting application (attempt {restart_count + 1})")
        
        # Run the app
        success = run_app(args.host, args.port, args.debug)
        
        if success:
            logger.info("Application exited successfully")
            break
        
        # Handle restart
        restart_count += 1
        if restart_count >= args.max_restarts:
            logger.error(f"Maximum restart attempts ({args.max_restarts}) reached. Exiting.")
            break
        
        logger.info(f"Restarting in {args.restart_delay} seconds...")
        time.sleep(args.restart_delay)

if __name__ == '__main__':
    main() 