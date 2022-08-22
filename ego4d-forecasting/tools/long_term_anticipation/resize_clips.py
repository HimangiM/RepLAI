import json
import sys
import tempfile
import av
import shutil
import subprocess
import os

ANNO_DIR = 'data/long_term_anticipation/annotations'

FULL_SCALE_DIR = 'data/long_term_anticipation/clips_hq'
# FULL_SCALE_DIR = '/Users/morgado/datasets/ego4d_data/v1/full_scale'

LOW_RES_DIR = 'data/long_term_anticipation/clips'
# LOW_RES_DIR = '/Users/morgado/datasets/ego4d_data/v1/clips_lr'

TEMP_DIR = '/compute/grogu-1-19/pmorgado/datasets/ego4d/temp'
# TEMP_DIR = '/Users/morgado/datasets/ego4d_data/v1/temp'

os.makedirs(LOW_RES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


VIDEO_FPS = 16
VIDEO_BLOCK_SIZE = 16
AUDIO_FPS = 44100
VIDEO_SIZE = 320


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def resize(clip_uid):
    src_fn = f"{FULL_SCALE_DIR}/{clip_uid}.mp4"
    dst_fn = f"{LOW_RES_DIR}/{clip_uid}.mp4"
    if os.path.isfile(dst_fn):
        print(bcolors.OKGREEN + f'Already processed: {src_fn}' + bcolors.ENDC)
        return

    if not os.path.isfile(src_fn):
        print(bcolors.OKBLUE + f'Does not exist: {src_fn}' + bcolors.ENDC)
        return

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR) as tmp_dir:
        ctr = av.open(src_fn)
        h = ctr.streams.video[0].codec_context.coded_height
        w = ctr.streams.video[0].codec_context.coded_width
        tmp_fn = f"{tmp_dir}/{clip_uid}.mp4"
        command = ['ffmpeg', '-y',
                   '-i', src_fn,
                   '-c:v', 'libx264',
                   '-crf', '28',
                   '-g', str(VIDEO_BLOCK_SIZE),
                   '-ar', str(AUDIO_FPS),
                   '-vf', f'scale="-2:\'min({VIDEO_SIZE},ih)\'"' if w > h else f'scale="\'min({VIDEO_SIZE},iw)\':-2"',
                   '-threads', '2',
                   '-loglevel', 'panic',
                   tmp_fn]
        os.system(' '.join(command))
        shutil.move(tmp_fn, dst_fn)
        print(bcolors.OKGREEN + f'Success: {src_fn}' + bcolors.ENDC)


def main():
    meta_train = json.load(open(f'{ANNO_DIR}/fho_lta_train.json'))
    meta_val = json.load(open(f'{ANNO_DIR}/fho_lta_val.json'))
    clip_uids = sorted(list(set(
        [clip['clip_uid'] for clip in meta_train['clips']] + [clip['clip_uid'] for clip in meta_val['clips']]
    )))

    import multiprocessing as mp
    pool = mp.Pool(20)
    pool.map(resize, clip_uids)

    # for clip in clip_uids:
    #     resize(clip)


if __name__ == '__main__':
    main()

