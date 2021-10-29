import numpy as np
import os

from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import RUNS_FOLDER
from utils.make_video import make_video_ffmpeg


if __name__ == '__main__':
    make_single_video = True
    batch_make_videos = False  # make videos in all runs subdirs

    if make_single_video:
        basedir = RUNS_FOLDER + os.sep + "explore" + os.sep + 'replot'
        source_dir = "vidW1"
        #fhead = "composite_lattice_step"
        fhead = "scatter_X_"
        ftype = ".jpg"
        nmax = 100
        fps = 2
        sourcepath = basedir + os.sep + source_dir
        outpath = basedir + os.sep + 'movie_slide4.mp4'
        make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)

    if batch_make_videos:
        overlapref = False
        basedir = RUNS_FOLDER + os.sep + "Annotated Multicell Data  Feb 2019" + os.sep + "other2"
        if overlapref:
            flagmod = " (overlapRef)"
            source_dir = "lattice" + os.sep + "overlapRef_0_0"
            fhead = "lattice_overlapRef_0_0"
        else:
            flagmod = ""
            source_dir = "lattice"
            fhead = "composite_lattice_step"
        ftype = ".png"
        nmax = 20
        fps = 1
        dirnames = os.listdir(basedir)
        rundirs = [os.path.join(basedir, o) for o in dirnames if os.path.isdir(os.path.join(basedir, o))]
        for idx, run in enumerate(rundirs):
            sourcepath = run + os.sep + source_dir
            outpath = run + os.sep + '%s%s.mp4' % (dirnames[idx], flagmod)
            make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)
