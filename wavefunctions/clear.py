import os
import shutil

current_file = os.path.abspath(__file__)
directory = os.path.dirname(current_file)

for item in os.listdir(directory):
    path = os.path.join(directory, item)
    if path != current_file:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

print("folder cleared.")