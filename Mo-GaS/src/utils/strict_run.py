import inspect
import os
import sys

frame_stack = inspect.stack()
in_notebook = False
for frame in frame_stack[::-1]:
  # print(frame.filename)
  if frame.filename[0] != '<':
    if 'ipykernel' in frame.filename:
      in_notebook = True
      # print('Running in a notebook', frame.filename)
      break

def ASSERT_NOT_RUN(_name_, _file_=None, _exit_=True, extra_msg=None):
  if _name_ == '__main__':
    print(f"This script{f' ({os.path.relpath(_file_, os.getcwd())})' if _file_ is not None else ''} is not meant to be run directly")
    if extra_msg is not None:
      print(extra_msg)
    if _exit_:
      sys.exit(1)

ASSERT_NOT_RUN(__name__, __file__)

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
  if in_notebook:
    os.chdir(folder)
  else:
    print(f"Please run this script from the root of the Mo-GaS project ({relative_path})")
    sys.exit(1)

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(main_file)))
