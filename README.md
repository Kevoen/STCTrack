# Don't Forget the Past, Learn From It for Object Tracking
This is the official implementation of the paper: Don't Forget the Past, Learn From It for Object Tracking.

![](https://picgo-1304301001.cos.ap-nanjing.myqcloud.com/PicGO/Architecture3.png)

## Setup
* Prepare Anaconda, CUDA and the corresponding toolkits. CUDA version required: 10.0+

* Create a new conda environment and activate it.
```Shell
conda create -n STCTrack python=3.7 -y
conda activate STCTrack
```

* Install `pytorch` and `torchvision`.
```Shell
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
# pytorch v1.5.0, v1.6.0, or higher should also be OK. 
```

* Install other required packages.
```Shell
pip install -r requirements.txt
```

## Test datasets format
* Prepare the datasets: OTB2015, VOT2018, UAV123, GOT-10k, TrackingNet, LaSOT, ILSVRC VID*, ILSVRC DET*, COCO*, and something else you want to test. Set the paths as the following: 
```Shell
├── STCTrack
|   ├── ...
|   ├── ...
|   ├── datasets
|   |   ├── COCO -> /opt/data/COCO
|   |   ├── GOT-10k -> /opt/data/GOT-10k
|   |   ├── ILSVRC2015 -> /opt/data/ILSVRC2015
|   |   ├── LaSOT -> /opt/data/LaSOT/LaSOTBenchmark
|   |   ├── OTB
|   |   |   └── OTB2015 -> /opt/data/OTB2015
|   |   ├── TrackingNet -> /opt/data/TrackingNet
|   |   ├── UAV123 -> /opt/data/UAV123/UAV123
|   |   ├── VOT
|   |   |   ├── vot2018
|   |   |   |   ├── VOT2018 -> /opt/data/VOT2018
|   |   |   |   └── VOT2018.json
```
* Notes

> i. Star notation(*): just for training. You can ignore these datasets if you just want to test the tracker.
> 
> ii. In this case, we create soft links for every dataset. The real storage location of all datasets is `/data/dir/`. You can change them according to your situation.
> 
> iii. The `VOT2018.json` file can be download from [here](https://drive.google.com/file/d/15iXOqZhPAJ-EnaMTLUsJkwMsUCneUq4V/view?usp=sharing).


* Download the models we trained.
    
  - [GOT-10k model](😀) (😁Coming soon...)
  - [fulldata model](😀) (😁Coming soon...)


* Use the path of the trained model to set the `pretrain_model_path` item in the configuration file correctly, then run the shell command.

* Note that all paths we used here are relative, not absolute. See any configuration file in the `experiments` directory for examples and details.

### General command format
```Shell
python main/test.py --config testing_dataset_config_file_path
```

Take GOT-10k as an example:
```Shell
python main/test.py --config experiments/stctrack/test/got10k/stctrack-got.yaml
```

## Training
* Prepare the datasets as described in the last subsection.
* Download the pretrained backbone model from [here](😀)( 😁Coming soon...).
* Run the shell command.

### training based on the GOT-10k benchmark
```Shell
python main/train.py --config experiments/stctrack/train/got10k/stctrack-trn.yaml
```

### training with full data
```Shell
python main/train.py --config experiments/stmtrack/train/fulldata/stctrack-trn-fulldata.yaml
```

## Testing Results
Click [Baidu Web Drive](https://pan.baidu.com/s/1rOdLpDLqNF5aZYAgbGQOzA?pwd=0fzv)(code:0fzv) to download all the following.
* OTB2015: [Baidu Web Drive](https://pan.baidu.com/s/1zXiTojI5IxyPGd-kInq_Ig?pwd=z856) code:z856 , [Google Drive]()(😁Coming soon...)
* GOT-10k:[Baidu Web Drive](https://pan.baidu.com/s/1M6K03augsRXPAkEb8Chfeg?pwd=xnwq) code:xnwq , [Google Drive]()(😁Coming soon...)
* LaSOT:[Baidu Web Drive](https://pan.baidu.com/s/1Ri6ZjM1m9nUnyCKnNECpUA?pwd=6zfu) code:6zfu , [Google Drive]()(😁Coming soon...)
* TrackingNet:[Baidu Web Drive](https://pan.baidu.com/s/1WzrT0T6rPoBUqQd1rOkmEw?pwd=26at) code:26at , [Google Drive]()(😁Coming soon...)
* UAV123:[Baidu Web Drive](https://pan.baidu.com/s/10amO6XSlNllMJyYmlo773A?pwd=k3id) code:k3id , [Google Drive]()(😁Coming soon...)


## Acknowledgement
### Repository

* [video_analyst](https://github.com/MegviiDetection/video_analyst)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytracking](https://github.com/visionml/pytracking)
* [PySOT](https://github.com/STVIR/pysot)
* [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)


This repository is developed based on the single object tracking framework [video_analyst](https://github.com/MegviiDetection/video_analyst). See it for more instructions and details.


## References
```Bibtex
None
```

## Contact
* Kai Huang[@kevoen](https://github.com/Kevoen)

If you have any questions, just create issues or [email](huangkai_edu@163.com) me:smile:.
