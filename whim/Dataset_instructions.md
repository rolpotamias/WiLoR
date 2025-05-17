## WHIM Dataset

**Annotations** 

The image annotations can be downloaded from the following Dropbox:

```
https://www.dropbox.com/scl/fo/ukvdpfjpagd5fmymfk268/AFm1t47rpCSxVrZvftZOaUA?rlkey=iixr3oea0u4et4mm4cjfldiia&st=bg7r369r&dl=0
```

**Images** 

To download the corresponding images you need to first download the YouTube videos and extract the specific frames.
You will need to install ''pytubefix'' or any similar package to download YouTube videos: 
```
pip install -Iv pytubefix==8.12.2
```
You can then run the following command to download the corresponding train/test images: 
```
python download_videos.py --mode {train/test}
```
Please make sure that the data are downloaded in the same directory. 