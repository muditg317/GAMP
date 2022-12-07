import inspect
import os
import sys

if __name__ == '__main__':
  print('This script is not meant to be run directly')
  sys.exit(1)

if 'Mo-GaS' not in os.getcwd().split(os.sep):
  print('Please run this script from the root of the project (Mo-GaS)')
  sys.exit(1)

# get the name of the file that is importing this script
main_file = None
for frame in inspect.stack()[::-1]:
  # print(frame.filename)
  if frame.filename[0] != '<':
    main_file = frame.filename
    break

# enforce that the current working directory is the root of the project
if os.getcwd().split(os.sep)[-1] != 'Mo-GaS':
  folder = os.getcwd()
  while folder.split(os.sep)[-1] != 'Mo-GaS':
    folder = os.path.dirname(folder)
  relative_path = os.path.relpath(folder, os.getcwd())
  print(f"Please run this script from the root of the Mo-GaS project ({relative_path})")
  sys.exit(1)

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(main_file)))
