# Video Frame Interpolation (RAFT + Softsplat)

put the following pre-trained weights into dir 'models':

[RAFT](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

[SoftSplat](https://drive.google.com/file/d/1_x_3CxY1_f83spqHt5s-ZwsEtNCTULLy/view?usp=sharing)

# Usage:

## Video Frame Interpolation

you need to spilt video into frames in dir 'input' using ffmpeg

for 4x interpolation:

  python interpolate_video.py --scale=0.5 --times=4

recommend: scale=0.5 when input frames resoultion is 1080p, scale=1.0 when < 1080p

## Image Interpolation

python interpolate_img.py


# Reference:
[RAFT](https://github.com/princeton-vl/RAFT)  

[SoftSplat-Full](https://github.com/JHLew/SoftSplat-Full)
