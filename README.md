# dehazeProcessor
> A project that, at present, offers two methods for image or video dehazing.
---
> ## TWO MAIN CLASSES:
> * darkchannelPriorProcessor
> * nonLocalDehazeProcessor

## darkchannelPriorProcessor
> implement of the paper [Single Image Haze Removal Using Dark Channel Prior](https://www.ncbi.nlm.nih.gov/pubmed/20820075)

## nonLocalDehazeProcessor
> implement of the paper [Non-local Image Dehazing](http://ieeexplore.ieee.org/document/7780554/?arnumber=7780554)

## build

    mkdir build
    cd build
    cmake ..
    make

## test on images

    ./darkchannel <img>       # a single image
    ./darkchannel <input-dir> <output-dir>  # a foler of images

## test on video

    ./darkchannel <input-video> <output-video>

## REFERNCES
[A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior](http://ieeexplore.ieee.org/abstract/document/7128396/)

[Single Image Haze Removal Using White Balancing and Saliency Map](http://www.sciencedirect.com/science/article/pii/S1877050915000435)

[The Next Best Underwater View](http://ieeexplore.ieee.org/document/7780778/)

[Optimized contrast enhancement for real-time image and video dehazing](http://www.sciencedirect.com/science/article/pii/S1047320313000242)
