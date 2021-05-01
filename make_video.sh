#!/bin/sh

ffmpeg -r 30 -f image2 -s 1000x1000 -i images/%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p vines.mp4