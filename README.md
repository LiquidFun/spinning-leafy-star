# Spinning Leafy Star

A Python script to create the following animation:

![](media/vines.gif)

It uses bezier curves to create the default "hook" shape. This shape is then duplicated by a basic transform and copied 5 times over. The animation is created by drawing all individual images and then combining them with ffmpeg. The variations between each cycle are created by varying the angle in which each child hook appears and changing whether the hook is flipped by the x-axis or by the y-axis.


## Create an Animation

Install requirements:

```
pip3 install -r requirements.txt
```

Run the animation script:

```
python3 animation.py
```

Combine into video and into gif:

```
./make_video.sh
```
