from src.utils.config import *
import os
from tqdm import tqdm
from subprocess import call
import h5py
import torch
from src.data.loaders import load_action_data, load_gaze_data
from src.features.feat_utils import transform_images, fuse_gazes_noop, reduce_gaze_stack, fuse_gazes, compute_motion
from src.utils.config import *

from src.data.utils import get_game_entries_, process_gaze_data  # nopep8



def create_interim_files(game='breakout'):
    """ Reads the directories in src/data/raw directory and creates interfim files in 
        src/data/interim directory
        Unzips the compressed files and creates gaze data files

        Args:
        ----
            game -- name of the game

        Returns:
        ----
            None
    """
    valid_actions = VALID_ACTIONS[game]
    game_runs, game_runs_dirs, game_runs_gazes = get_game_entries_(
        os.path.join(RAW_DATA_DIR, game))

    interim_game_dir = os.path.join(INTERIM_DATA_DIR, game)
    if not os.path.exists(interim_game_dir):
        os.makedirs(interim_game_dir)

    for game_run, game_run_dir, game_run_gaze in tqdm(
            zip(game_runs, game_runs_dirs, game_runs_gazes)):
        untar_sting = 'tar -xjf {} -C {}'.format(
            os.path.join(game_run_dir, game_run) + CMP_FMT,
            interim_game_dir + '/')
        untar_args = untar_sting.split(' ')
        interim_writ_dir = os.path.join(interim_game_dir, game_run)
        gaze_out_file = '{}/{}_gaze_data.csv'.format(interim_writ_dir,
                                                     game_run)

        if os.path.exists(os.path.join(interim_game_dir, game_run)):
            print("Exists, Skipping {}/{}".format(game_run_dir, game_run))
        else:
            print("Extracting {}/{}".format(game_run_dir, game_run))
            call(untar_args)

        if not os.path.exists(gaze_out_file) or OVERWRITE_INTERIM_GAZE:
            print("Prepping gaze data for {}/{}".format(
                game_run_dir, game_run))
            gaze_file = os.path.join(game_run_dir, game_run_gaze)
            process_gaze_data(gaze_file, gaze_out_file, valid_actions)
        else:
            print("Exists, Skipping prepping of {}/{}".format(
                game_run_dir, game_run))


def create_processed_data(stack=1,
                          stack_type='',
                          stacking_skip=1,
                          from_ix=0,
                          till_ix=-1,
                          game='breakout',
                          data_types=['images', 'actions', 'gazes'],
                          recompute=[],
                          bad_keys=['motion']):
    """ Loads data from all the game runs in the src/data/interim  directory, and 
        creates a hdf file in the src/data/processed directory.
        The hdf file contains a dataset per run per data type.
    
        Args:
        ----
            data_types -- types of data to save, contains atleast on of the following
                        ['images', 'actions', 'gazes', 'fused_gazes', 'gazes_fused_noop']

            stack -- number of frames in the stack

            stacking_skip -- Number of frames to skip while stacking
            
            from_ix --  starting index in the data, default is first, 0
            
            till_ix -- last index of the the data to be considered, default is last ,-1

            game : game to load the data from, directory of game runs
    
        Returns:
        ----
            None
    """

    game_dir = os.path.join(INTERIM_DATA_DIR, game)
    game_runs = os.listdir(game_dir)
    images = []
    actions = []
    game_h5_filename = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    game_h5_file = h5py.File(game_h5_filename, 'a')

    for game_run in tqdm(game_runs):
        if game_run not in game_h5_file.keys():
            print(f"Creating processed data for [{game} - {game_run}]")
            group = game_h5_file.create_group(game_run)
        else:
            print(f"Some processed data for [{game} - {game_run}] already exists\n\t{game_h5_file[game_run].keys()}")
            group = game_h5_file[game_run]

        do_images = 'images' in data_types and ('images' not in group.keys() or 'images' in recompute)
        do_actions = 'actions' in data_types and ('actions' not in group.keys() or 'actions' in recompute)
        do_gazes = 'gazes' in data_types and ('gazes' not in group.keys() or 'gazes' in recompute)
        do_fused_gazes = 'fused_gazes' in data_types and ('fused_gazes' not in group.keys() or 'fused_gazes' in recompute)
        do_gazes_fused_noop = 'gazes_fused_noop' in data_types and ('gazes_fused_noop' not in group.keys() or 'gazes_fused_noop' in recompute)
        # do_motion = 'motion' in data_types and ('motion' not in group.keys() or 'motion' in recompute)

        if do_images or do_actions or do_gazes_fused_noop:
            print(f"\tLoading images and actions")
            images_, actions_ = load_action_data(stack, stack_type, stacking_skip,
                                                from_ix, till_ix, game=game, game_run=game_run)
            print(f"images: stacks of shape {len(images_)} x {len(images_[0])} x {images_[0][0].shape}")

        if do_images:
            print(f"\t\tSaving images dataset")
            images_data = transform_images(images_, type='torch')
            images_data = images_data.cpu().numpy()
            
            if 'images' in group.keys():
                del group['images']
            group.create_dataset('images',
                                data=images_data,
                                compression=config_data['HDF_CMP_TYPE'],
                                compression_opts=config_data['HDF_CMP_LEVEL'])
            
            del images_data
        if do_actions:
            print(f"\t\tSaving actions dataset")

            if 'actions' in group.keys():
                del group['actions']
            group.create_dataset('actions',
                                data=actions_,
                                compression=config_data['HDF_CMP_TYPE'],
                                compression_opts=config_data['HDF_CMP_LEVEL'])

        if do_gazes or do_gazes_fused_noop:
            print(f"\tLoading gazes")
            _, gazes = load_gaze_data(stack,
                                    stack_type,
                                    stacking_skip,
                                    from_ix,
                                    till_ix,
                                    game=game,
                                    game_run=game_run,
                                    skip_images=True)

            if do_gazes_fused_noop:
                print(f"\t\tSaving fused (noop) gazes dataset")
                gazes_fused_noop = fuse_gazes_noop(images_,
                                                gazes,
                                                actions_,
                                                gaze_count=1,
                                                fuse_type='stack',
                                                fuse_val=0)

                gazes_fused_noop = gazes_fused_noop.numpy()

                if 'gazes_fused_noop' in group.keys():
                    del group['gazes_fused_noop']
                group.create_dataset('gazes_fused_noop',
                                    data=gazes_fused_noop,
                                    compression=config_data['HDF_CMP_TYPE'],
                                    compression_opts=config_data['HDF_CMP_LEVEL'])

                del gazes_fused_noop

            if do_gazes:
                print(f"\t\tSaving gazes dataset")

                gazes = torch.stack(
                    [reduce_gaze_stack(gaze_stack) for gaze_stack in gazes])
                gazes = gazes.numpy()

                if 'gazes' in group.keys():
                    del group['gazes']
                group.create_dataset('gazes',
                                    data=gazes,
                                    compression=config_data['HDF_CMP_TYPE'],
                                    compression_opts=config_data['HDF_CMP_LEVEL'])

            del gazes

        # if do_motion:
        #     print(f"\tComputing motion")

        #     motion = compute_motion(images_)

        #     print(f"\t\tSaving motion dataset")
        #     motion = motion.cpu().numpy()

        #     if 'motion' in group.keys():
        #         del group['motion']
        #     group.create_dataset('motion',
        #                         data=motion,
        #                         compression=config_data['HDF_CMP_TYPE'],
        #                         compression_opts=config_data['HDF_CMP_LEVEL'])

        #     del motion
        
        for key in bad_keys:
            if key in group.keys():
                print(f"\t\tDeleting bad key: {key}")
                del group[key]

        if do_images or do_actions or do_gazes_fused_noop:
            del images_, actions_

    game_h5_file.close()


def combine_processed_data(game):
    """ Reads the specified hdf5 file, and combines all the groups into a
        single combined group in the same file.
        For each data type, the data is concatenated between different runs.

        Args:
        ----
            game -- name of the hdf5 file to combine, assumed to be in processed directory, without the extension

        Returns:
        ----
            None
    """

    game_h5_filename = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    game_h5_file = h5py.File(game_h5_filename, 'a')

    groups = list(game_h5_file.keys())
    if not 'combined' in groups:
        all_group = game_h5_file.create_group('combined')
    all_group = game_h5_file['combined']
    data = list(game_h5_file[groups[0]].keys())

    for datum in tqdm(data):
        if datum in all_group.keys():
            print(f"Combined {datum} values for {game} already exist, skipping")
            continue
        max_shape_datum = (sum([
            game_h5_file[group][datum].shape[0] for group in groups
            if group != 'combined'
        ]), *game_h5_file[groups[0]][datum].shape[1:])
        print(f"Combining {game}: {max_shape_datum} {datum} values")
        all_group.create_dataset(
            datum,
            data=game_h5_file[groups[0]][datum][:],
            maxshape=max_shape_datum,
            compression=config_data['HDF_CMP_TYPE'],
        )

        for group in tqdm(groups[1:]):
            game_h5_file['combined'][datum].resize(
                game_h5_file['combined'][datum].shape[0] +
                game_h5_file[group][datum].shape[0],
                axis=0)
            game_h5_file['combined'][datum][
                game_h5_file['combined'][datum].
                shape[0]:, :] = game_h5_file[group][datum]

    game_h5_file.close()


def remove_combined_data(game):
    """ Reads the specified hdf5 file, and deletes the combined group if present.

        Args:
        ----
            game -- name of the hdf5 file for which to remove combinations, assumed to be in processed directory, without the extension

        Returns:
        ----
            None
    """

    game_h5_filename = os.path.join(PROC_DATA_DIR, game + '.hdf5')
    game_h5_file = h5py.File(game_h5_filename, 'a')

    groups = list(game_h5_file.keys())
    if 'combined' in groups:
        del game_h5_file['combined']


if __name__ == "__main__":
    for game in GAMES_FOR_TRAINING:
        try:
            print(f"Processing {game}")
            processed_file = os.path.join(PROC_DATA_DIR, game + '.hdf5')
            if not os.path.exists(processed_file):
                create_interim_files(game=game)
            create_processed_data(stack=STACK_SIZE,
                                game=game,
                                till_ix=-1,
                                stacking_skip=1,
                                data_types=DATA_TYPES)
        except BlockingIOError as e:
            print(f"BlockingIOError: {e}")
            print(f"Skipping {game}")
        # combine_processed_data(game)
        # remove_combined_data(game)
