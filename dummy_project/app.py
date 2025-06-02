# This is a dummy Python file for testing Guardian.
import os
import json # Adding some imports for testing
import datetime # This will be unused

def hello_world():
    print(f"Hello, Guardian! Current OS: {os.name}")
    my_data = {"key": "value"}
    print(json.dumps(my_data))
    # today = datetime.date.today() # Commenting out usage

if __name__ == "__main__":
    hello_world()