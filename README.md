# HDS-SAM

This is the official implementation of [HDS-SAM: Hybrid Dual-Stream Segment Anything Model with Volume-Sphericity Weighting Strategy for Small Tumor Segmentation] at AAAI-26.

## 🌟 Highlights

-  We propose an innovative HDS-SAM method, which introduces the HDSE that effectively combines the strengths of Transformer and U-Net architectures to comprehensively extract both global anatomical features and local texture features, enabling efficient segmentation of small tumors.
-  We design a triple axial pooling strategy tailored for both the Transformer and modified U-Net encoders, which enhances spatial information preservation, reduces parameter usage, and significantly improves the representation of small tumor regions.
-  We develop a novel weighting strategy, adaptively adjusting the loss function based on tumor volume and sphericity, improving the model's sensitivity to small and irregular tumor structures.
-  Extensive experiments conducted on both in-house and public datasets validate the effectiveness and generalization capability of HDS-SAM.

## 📦 Requirement

Run the following command to install the required packages:

 ```bash
pip install -r requirements.txt
 ```

Then download [SAM checkpoint](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view), and put it at ./ckpt

## 🔨 Usage

### 1. Pre-processing

Prepare your own train dataset and adjust the corresponding code according to your specific scenario (using the LiTS dataset as an example).

Run the following command to pre-process the dataset:

```angular2
python ./utils/LiTSProcess.py
```

### 2. Training

Now you can start to train the HDS-SAM:

```angular2
python train_lits.py
```

### 3. Test

You can also test our HDS-SAM directly:

```angular2
python validation_lits.py
```

## 🙏 Acknowledgement

The code is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D).

We thank the authors for their open-sourced code and encourage users to cite their works when applicable.