# Computer Vision Project

## Requirements

- [TQDM](https://github.com/tqdm/tqdm) - Progress Bar
- [CV2](https://pypi.org/project/opencv-python/) - cv2
- [Numba](http://numba.pydata.org/) - CUDA
- [Source Images](https://sourceforge.net/projects/adobedatasets.adobe/files/adobe_panoramas.tgz/download)
- SURF using xfeatures2d

---

## How to run

> python index.py [-f [FOLDER]] [--crop] [--debug] [-t --type [IMG TYPE]]

> python index.py (default folder files\\)

---

## Explanation

    We take a row of images from img_0 to img_n. We find the middle index and image
    then preprocess the data into two lists. A left side and right side. For the left
    side, we find and computer features using SIFT and BF. We inverse the homography
    so we can run homography on the left side towards the center. After that is
    completed, we do normal homography and pass right side images into the half
    constructed panorama which causes lag and slowdown of the system. We substitute
    this by using CUDA to speed up the math and stitching.

---

## Image by Image Steps

- Image 1
![Img1](https://github.com/Nomulous/ComputerVisionFinal/blob/master/files/goldengate-00.png)
- Image 2
![Img2](https://github.com/Nomulous/ComputerVisionFinal/blob/master/files/goldengate-01.png)
- Matches ![Matches](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/matches.png)
- Stitched ![Stitched](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/goldengate-00_2_pano.png)

---

## Credits

- [Martin Tuzim](https://github.com/nomulous) - Code

---

## Sample Images

![GoldenGate](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/goldengate-00_pano.png)

![Rio](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/rio-00_pano.png)

![Lunch Room](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/img01_pano.png)

![Half Dome](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/halfdome-00_pano.png)

![Diamond Head](https://github.com/Nomulous/ComputerVisionFinal/blob/master/img/diamondhead-00_pano.png)
