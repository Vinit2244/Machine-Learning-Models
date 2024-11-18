<center>

# **Assignment 5 Report**

</center>

## **Table of Contents**
2. [KDE](#KDE)
  - [Synthetic Data Generation](#SyntheticData)
  - [KDE vs GMM](#KDEvsGMM)
3. [HMM](#HMM)
  - [MFCC Feature Spectrogram](#MFCCSpectrogram)
  - [Comparing Model's Performance](#Compare)
4. [RNN](#RNN)
  - [Counting Bits](#CountingBits)
  - [OCR](#OCR)

---

<p id = "KDE"> </p>

<p id = "SyntheticData"> </p>

## **2.2 Generating Synthetic Data**

<center>

![Generated Synthetic Data Plot](./figures/synthetic_data.png)

*Figure 1: Synthetic Data Generated*

</center>

An **outer bigger disk** centered at **(0, 0)**, radius **2** and noise **0.2** with 3000 points and a **inner smaller disk** centered at **(1, 1)**, radius **0.25** and noise **0.1** with 500 points.

> <span style="color: green;">Note: All the parameters have purely been judged merely by visual cues and are subject to errors.</span>

---

<p id="KDEvsGMM"></p>

## **2.3 KDE vs GMM**

### KDE

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/kde_contour.png" alt="KDE density function in 2D" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 2: KDE Density Function in 2D</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/kde_3d.gif" alt="KDE Density Function 3D GIF" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 3: KDE Density Function in 3D</p>
  </div>
</div>

> <span style="color: green;">We can see that the KDE model consistently fits the data with bandwith <b>0.5</b></span>

### GMM

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_contour_2.png" alt="GMM Density Plot 2D with k = 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 4: GMM Density Plot 2D with k = 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_contour_3.png" alt="GMM Density Plot 2D with k = 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 5: GMM Density Plot 2D with k = 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_contour_4.png" alt="GMM Density Plot 2D with k = 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 6: GMM Density Plot 2D with k = 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_contour_5.png" alt="GMM Density Plot 2D with k = 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 7: GMM Density Plot 2D with k = 5</p>
  </div>
</div>

> <span style="color: green;">We can see that as the number of clusters increases in GMM it tries to equally space out each cluster so that each cluster contains almost equal number of points and for k = 2 it is clearly able to identify 2 clusters one big and one small.</span>

### Best GMM (We know 2 clusters) k = 2

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_contour_2.png" alt="GMM Density Plot 2D with k = 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 8: GMM Density Plot 2D with k = 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/kde_vs_gmm/gmm_3d_2.gif" alt="GMM Density Function 3D GIF k = 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 9: GMM Density Function in 3D with k = 2</p>
  </div>
</div>

---

<p id="HMM"></p>

<p id="MFCCSpectrogram"></p>

## **3.2 MFCC Spectrogram for each digit and person**

### Digit 0

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_0_mfcc_heatmap.png" alt="George 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 10: George 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_0_mfcc_heatmap.png" alt="Jackson 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 11: Jackson 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_0_mfcc_heatmap.png" alt="Lucas 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 12: Lucas 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_0_mfcc_heatmap.png" alt="Nicolas 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 13: Nicolas 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_0_mfcc_heatmap.png" alt="Theo 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 14: Theo 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_0_mfcc_heatmap.png" alt="Yweweler 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 15: Yweweler 0</p>
  </div>
</div>

### Digit 1

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_1_mfcc_heatmap.png" alt="George 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 16: George 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_1_mfcc_heatmap.png" alt="Jackson 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 17: Jackson 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_1_mfcc_heatmap.png" alt="Lucas 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 18: Lucas 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_1_mfcc_heatmap.png" alt="Nicolas 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 19: Nicolas 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_1_mfcc_heatmap.png" alt="Theo 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 20: Theo 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_1_mfcc_heatmap.png" alt="Yweweler 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 21: Yweweler 1</p>
  </div>
</div>

### Digit 2

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_2_mfcc_heatmap.png" alt="George 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 22: George 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_2_mfcc_heatmap.png" alt="Jackson 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 23: Jackson 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_2_mfcc_heatmap.png" alt="Lucas 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 24: Lucas 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_2_mfcc_heatmap.png" alt="Nicolas 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 25: Nicolas 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_2_mfcc_heatmap.png" alt="Theo 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 26: Theo 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_2_mfcc_heatmap.png" alt="Yweweler 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 27: Yweweler 2</p>
  </div>
</div>

### Digit 3

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_3_mfcc_heatmap.png" alt="George 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 28: George 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_3_mfcc_heatmap.png" alt="Jackson 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 29: Jackson 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_3_mfcc_heatmap.png" alt="Lucas 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 30: Lucas 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_3_mfcc_heatmap.png" alt="Nicolas 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 31: Nicolas 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_3_mfcc_heatmap.png" alt="Theo 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 32: Theo 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_3_mfcc_heatmap.png" alt="Yweweler 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 33: Yweweler 3</p>
  </div>
</div>

### Digit 4

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_4_mfcc_heatmap.png" alt="George 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 34: George 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_4_mfcc_heatmap.png" alt="Jackson 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 35: Jackson 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_4_mfcc_heatmap.png" alt="Lucas 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 36: Lucas 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_4_mfcc_heatmap.png" alt="Nicolas 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 37: Nicolas 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_4_mfcc_heatmap.png" alt="Theo 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 38: Theo 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_4_mfcc_heatmap.png" alt="Yweweler 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 39: Yweweler 4</p>
  </div>
</div>

### Digit 5

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_5_mfcc_heatmap.png" alt="George 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 40: George 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_5_mfcc_heatmap.png" alt="Jackson 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 41: Jackson 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_5_mfcc_heatmap.png" alt="Lucas 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 42: Lucas 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_5_mfcc_heatmap.png" alt="Nicolas 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 43: Nicolas 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_5_mfcc_heatmap.png" alt="Theo 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 44: Theo 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_5_mfcc_heatmap.png" alt="Yweweler 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 45: Yweweler 5</p>
  </div>
</div>

### Digit 6

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_6_mfcc_heatmap.png" alt="George 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 46: George 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_6_mfcc_heatmap.png" alt="Jackson 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 47: Jackson 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_6_mfcc_heatmap.png" alt="Lucas 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 48: Lucas 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_6_mfcc_heatmap.png" alt="Nicolas 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 49: Nicolas 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_6_mfcc_heatmap.png" alt="Theo 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 50: Theo 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_6_mfcc_heatmap.png" alt="Yweweler 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 51: Yweweler 6</p>
  </div>
</div>

### Digit 7

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_7_mfcc_heatmap.png" alt="George 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 52: George 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_7_mfcc_heatmap.png" alt="Jackson 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 53: Jackson 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_7_mfcc_heatmap.png" alt="Lucas 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 54: Lucas 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_7_mfcc_heatmap.png" alt="Nicolas 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 55: Nicolas 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_7_mfcc_heatmap.png" alt="Theo 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 56: Theo 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_7_mfcc_heatmap.png" alt="Yweweler 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 57: Yweweler 7</p>
  </div>
</div>

### Digit 8

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_8_mfcc_heatmap.png" alt="George 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 58: George 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_8_mfcc_heatmap.png" alt="Jackson 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 59: Jackson 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_8_mfcc_heatmap.png" alt="Lucas 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 60: Lucas 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_8_mfcc_heatmap.png" alt="Nicolas 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 61: Nicolas 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_8_mfcc_heatmap.png" alt="Theo 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 62: Theo 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_8_mfcc_heatmap.png" alt="Yweweler 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 63: Yweweler 8</p>
  </div>
</div>

### Digit 9

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/george_digit_9_mfcc_heatmap.png" alt="George 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 64: George 9</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/jackson_digit_9_mfcc_heatmap.png" alt="Jackson 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 65: Jackson 9</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/lucas_digit_9_mfcc_heatmap.png" alt="Lucas 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 66: Lucas 9</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/nicolas_digit_9_mfcc_heatmap.png" alt="Nicolas 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 67: Nicolas 9</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/theo_digit_9_mfcc_heatmap.png" alt="Theo 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 68: Theo 9</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/mfcc_heatmaps/yweweler_digit_9_mfcc_heatmap.png" alt="Yweweler 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 69: Yweweler 9</p>
  </div>
</div>

### My own recordings

<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/0_vinit_mfcc_heatmap.png" alt="Vinit 0" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 70: Vinit 0</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/1_vinit_mfcc_heatmap.png" alt="Vinit 1" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 71: Vinit 1</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/2_vinit_mfcc_heatmap.png" alt="Vinit 2" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 72: Vinit 2</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/3_vinit_mfcc_heatmap.png" alt="Vinit 3" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 73: Vinit 3</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/4_vinit_mfcc_heatmap.png" alt="Vinit 4" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 74: Vinit 4</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/5_vinit_mfcc_heatmap.png" alt="Vinit 5" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 75: Vinit 5</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/6_vinit_mfcc_heatmap.png" alt="Vinit 6" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 76: Vinit 6</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/7_vinit_mfcc_heatmap.png" alt="Vinit 7" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 77: Vinit 7</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/8_vinit_mfcc_heatmap.png" alt="Vinit 8" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 78: Vinit 8</p>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img src="./figures/my_recordings_heatmaps/9_vinit_mfcc_heatmap.png" alt="Vinit 9" style="width: 100%; max-width: 500px;"/>
    <p style="text-align: center;">Figure 79: Vinit 9</p>
  </div>
</div>

> <span style="color: green;">A regular pattern can be observed in MFCC feature spectrogram for each digit which is consistent across different people for example in 0 we can see a flat region of high db at the ending of the clip, </span>

> <span style="color: green;">The present of consecutive patterns with occurence in a specific sequence one after the other explains as in why HMMs can be useful in this problem as it can learn based on the sequence of patterns observed the probabilities of each digit.</span>

---

<p id="Compare"></p>

## **3.4 Comparing Models Performance**

Output:
```
Accuracy on provided test set: 99.12%

Now testing on my own recordings
True Label: 0, Predicted Label: 9
True Label: 1, Predicted Label: 4
True Label: 2, Predicted Label: 4
True Label: 3, Predicted Label: 2
True Label: 4, Predicted Label: 2
True Label: 5, Predicted Label: 4
True Label: 6, Predicted Label: 0
True Label: 7, Predicted Label: 0
True Label: 8, Predicted Label: 1
True Label: 9, Predicted Label: 1
Accuracy on my own recordings: 0.0%
```

> <span style="color: green;">The model is not good at generalisation, it depends on how the audio is recorded and cropped. Due to my manual recording and cropping it might not match with the one in the dataset and hence we can say that the model perform poorly on my voice due to it's overfitting to the preprocessing done to the audio clips.</span>

---
<p id="RNN"></p>

## RNN

<p id="CountingBits"></p>

### Counting Bits

<p id="OCR"></p>

### OCR

> Both of their reports are in respective .ipynb files itself.