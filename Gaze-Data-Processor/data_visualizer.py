# data_visualizer.py
#
# Visualize the cleaned data (to verify the correctness)
# -----------------------
import os
import time
import shutil
import sys
# noinspection PyClassHasNoInit
import tarfile
import pygame
import data_reader

class DrawgcWrapper:
    def __init__(self):
        self.cursor = pygame.image.load('target.png')
        self.cursor_size = (self.cursor.get_width(), self.cursor.get_height())

    def draw_gc(self, screen, gaze_position):
        """ draw the gaze-contingent window on screen """
        region_top_left = (
            gaze_position[0] - self.cursor_size[0] // 2, gaze_position[1] - self.cursor_size[1] // 2)
        # Draws and shows the cursor content;
        screen.blit(self.cursor, region_top_left)


class DrawingStatus:
    def __init__(self):
        pass
    draw_many_gazes = 0
    cur_frame_id = 0
    total_frame = 0
    target_fps = 60
    pause = False
    terminated = False


DS = DrawingStatus()


def preprocess_and_sanity_check(png_files, frameid_list):
    has_warning = False

    # remove unrelated files
    for fname in png_files:
        if not fname.endswith(".png"):
            print(
                "Warning: %s is not a PNG file. Deleting it from the frames list" % fname)
            has_warning = True
            png_files.remove(fname)

    # check if each frame has its corresponding png file
    based_dir = png_files[0].split('/')[0]
    set_png_files = set(png_files)
    for frameid in frameid_list:
        png_fname = based_dir + '/' + frameid + '.png'
        if png_fname not in set_png_files:
            print("Warning: no corresponding png file for frame id %s." % frameid)
            has_warning = True

    if has_warning:
        print("There are warnings. Sleeping for 2 sec...")
        time.sleep(2)


def frameid_sort_key(frameid):
    return int(frameid.split('_')[2])


def check_gaze_range(pos_x, pos_y, w, h):
    """ Check if the gaze coordinate is within the screen """
    if pos_x < 0 or pos_x > w or pos_y < 0 or pos_y > h:
        return False
    return True


def event_handler_func():
    global DS

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                print("Fast-backward 5 seconds")
                DS.cur_frame_id -= 5 * DS.target_fps
            elif event.key == pygame.K_DOWN:
                print("Fast-forward 5 seconds")
                DS.cur_frame_id += 5 * DS.target_fps
            if event.key == pygame.K_LEFT:
                print("Moving to previous frame")
                DS.cur_frame_id -= 1
            elif event.key == pygame.K_RIGHT:
                print("Moving to next frame")
                DS.cur_frame_id += 1
            elif event.key == pygame.K_F3:
                p = float(
                    input("Seeking through the video. Enter a percentage in float: "))
                DS.cur_frame_id = int(p/100*DS.total_frame)
            elif event.key == pygame.K_SPACE:
                DS.pause = not DS.pause
            elif event.key == pygame.K_F9:
                DS.draw_many_gazes = not DS.draw_many_gazes
                print("draw all gazes belonging to a frame: %s" %
                      ("ON" if DS.draw_many_gazes else "OFF"))
            elif event.key == pygame.K_F11:
                DS.target_fps -= 2
                print("Setting target FPS to %d" % DS.target_fps)
            elif event.key == pygame.K_F12:
                DS.target_fps += 2
                print("Setting target FPS to %d" % DS.target_fps)
            elif event.key == pygame.K_ESCAPE:
                DS.terminated = True
                print("\nStopping replay.")
    DS.cur_frame_id = max(0, min(DS.cur_frame_id, DS.total_frame - 1))
    DS.target_fps = max(1, DS.target_fps)


def visualize_csv(tar_fname, csv_fname):
    # read from the csv file
    frameid2pos, _, frameid2duration, _, _, _, frameid_list = data_reader.read_gaze_data_csv_file(
        csv_fname)
    frameid_list = sorted(frameid_list, key=frameid_sort_key)

    # open the tar file
    tar = tarfile.open(tar_fname, 'r')
    png_files = tar.getnames()

    preprocess_and_sanity_check(png_files, frameid_list)

    print("\nYou can control the replay using keyboard. Try pressing space/up/down/left/right.")
    print("For all available keys, see event_handler_func() code.\n")
    temp_extract_dir = os.path.dirname(tar_fname) + "/gaze_data_tmp/"
    if not os.path.exists(temp_extract_dir):
        os.mkdir(temp_extract_dir)

    print("Uncompressing PNG tar file")
    if not os.path.exists(temp_extract_dir+'/'+png_files[0].split('/')[0]) or len(os.listdir(temp_extract_dir+'/'+png_files[0].split('/')[0])) != len(png_files):
        if os.path.exists(temp_extract_dir+'/'+png_files[0].split('/')[0]):
            print(len(os.listdir(temp_extract_dir+'/'+png_files[0].split('/')[0])), len(png_files))
        tar.extractall(temp_extract_dir)

    # get the full path
    temp_extract_full_path_dir = temp_extract_dir + \
        '/' + png_files[0].split('/')[0]
    print("Uncompressed PNG tar file into temporary directory: " +
          temp_extract_full_path_dir)

    # init pygame and other stuffs
    origin_w = 160
    x_scale = 3.0
    origin_h = 210
    y_scale = 3.0
    w, h = int(origin_w*x_scale), int(origin_h*y_scale)

    global DS
    DS.target_fps = 60
    DS.total_frame = len(png_files)
    DS.cur_frame_id = 0
    DS.terminated = False

    # init Drawgc Wrapper
    dw = DrawgcWrapper()

    # init pygame
    pygame.init()
    pygame.font.init()
    pygame_font = pygame.font.SysFont('Consolas', 28)
    pygame.display.set_mode((w, h), pygame.RESIZABLE |
                            pygame.DOUBLEBUF | pygame.RLEACCEL, 32)
    screen = pygame.display.get_surface()

    while True:  # DS.cur_frame_id < DS.total_frame:
        event_handler_func()

        # Load PNG file and draw the frame and the gaze-contingent window
        frame_id = frameid_list[DS.cur_frame_id]
        png_fname = temp_extract_full_path_dir + '/' + frame_id + '.png'
        # check if the corresponding png file exists
        if not os.path.isfile(png_fname):
            screen.fill((0, 0, 0))
            text_surface_desc = pygame_font.render(
                'Missing png file for frame id:', True, (255, 255, 255))
            screen.blit(text_surface_desc, (w // 10, 2 * h // 5))
            text_surface_frameid = pygame_font.render(
                frame_id, True, (255, 255, 255))
            screen.blit(text_surface_frameid, (w // 10, 3 * h // 5))
        else:
            s = pygame.image.load(png_fname)
            s = pygame.transform.scale(s, (w, h))
            screen.blit(s, (0, 0))

            # visualize the gaze
            gaze_list = frameid2pos[frame_id]
            if gaze_list is not None and len(gaze_list) > 0:
                for (posX, posY) in gaze_list:
                    if check_gaze_range(posX, posY, origin_w, origin_h):
                        dw.draw_gc(screen, (posX*x_scale, posY*y_scale))

        pygame.display.flip()

        if not DS.pause:
            DS.cur_frame_id = DS.cur_frame_id + 1
        if DS.terminated:
            break

        duration = frameid2duration[frame_id]
        if duration is not None:
            time.sleep(duration * 0.001)  # duration is in msec

    print("Replay ended.")

    # remove the temporary files
    # print("Deleting PNG files in temporary directory.")
    # shutil.rmtree(temp_extract_full_path_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python data_visualizer.py tar_fname csv_fname')
        exit(1)

    tar_fname = sys.argv[1]

    if len(sys.argv) == 2:
        csv_fname = sys.argv[1].split(".tar")[0] + ".txt"
    elif len(sys.argv) == 3:
        csv_fname = sys.argv[2]

    visualize_csv(tar_fname, csv_fname)
