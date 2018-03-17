# INTRODUCTION
> A project that, at present, offers two methods for image and video dehazing.
---
## TWO MAIN METHOD:
### darkchannelPriorProcessor
implement of the paper [Single Image Haze Removal Using Dark Channel Prior](https://www.ncbi.nlm.nih.gov/pubmed/20820075)

### nonLocalDehazeProcessor
implement of the paper [Non-local Image Dehazing](http://ieeexplore.ieee.org/document/7780554/?arnumber=7780554)

### SWITCH PROCESSING METHOD
You can switch your processing method to any one, for example:
```C++
	deHazeByNonLocalMethod(src, dst, "../TR_SPHERE_2500.txt");
	deHazeByDarkChannelPrior(src, dst);
```

## USAGE
### dehaze a single image
this command will make a process on a single image and display the result

./darkchannel --type=0 --input=../input/a.bmp 
![img0](./img/img0.png)

### dehaze a series of images
this command will make a process on all images in the given input-folder 
and generate the result in the output-folder

./darkchannel --type=0 --input=../input/ --output=../output/
![img1](./img/img1.png)
![img1-1](./img/img1-1.png)

### dehaze a vedio and display
this command will make a process on a vedio given and display the result

./darkchannel --type=1 --input=../data/breed.mp4
![img2](.img/img2.png)

### dehaze and write a vedio
this command will make a process on a vedio given and write the result
into the given output file

./darkchannel --type=1 --input=../data/breed.mp4 --output=./out.avi
![img4](./img/img4.png)


## REFERNCES
[A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior](http://ieeexplore.ieee.org/abstract/document/7128396/)

[Single Image Haze Removal Using White Balancing and Saliency Map](http://www.sciencedirect.com/science/article/pii/S1877050915000435)

[The Next Best Underwater View](http://ieeexplore.ieee.org/document/7780778/)

[Optimized contrast enhancement for real-time image and video dehazing](http://www.sciencedirect.com/science/article/pii/S1047320313000242)
