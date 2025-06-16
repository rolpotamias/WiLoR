## WHIM Dataset

**Annotations** 

The image annotations can be downloaded from the following Dropbox:

```
https://drive.google.com/drive/folders/1d9Fw7LfnF5oJuA6yE8T3xA-u9p6H5ObZ
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
