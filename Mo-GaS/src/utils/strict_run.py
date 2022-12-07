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
while os.getcwd().split(os.sep)[-1] != 'Mo-GaS':
  os.chdir('..')

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(main_file)))
