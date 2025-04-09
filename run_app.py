#!/usr/bin/env python
"""
Indian Sign Language Translator - Main Application Runner

This script runs the Flask web application with the Indian Sign Language Translator models.
"""

import os
import sys
import argparse

def main():
    """Main function to run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Indian Sign Language Translator')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # Import the Flask app
    from src.web.simple_app import app
    
    # Run the app
    print(f"Starting Indian Sign Language Translator on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 