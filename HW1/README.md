## Unet：U-Net: Convolutional Networks for Biomedical Image Segmentation目標檢測模型在Pytorch當中的實現
---

### 目錄
1. [倉庫更新 Top News](#倉庫更新)
2. [相關倉庫 Related code](#相關倉庫)
3. [性能情況 Performance](#性能情況)
4. [所需環境 Environment](#所需環境)
5. [文件下載 Download](#文件下載)
6. [訓練步驟 How2train](#訓練步驟)
7. [預測步驟 How2predict](#預測步驟)
8. [評估步驟 miou](#評估步驟)
9. [參考資料 Reference](#Reference)

## Top News
**`2022-03`**:**進行大幅度更新、支持step、cos學習率下降法、支持adam、sgd優化器選擇、支持學習率根據batch_size自適應調整。**  
BiliBili視頻中的原倉庫地址為：https://github.com/bubbliiiing/unet-pytorch/tree/bilibili

**`2020-08`**:**創建倉庫、支持多backbone、支持數據miou評估、標註數據處理、大量註釋等。**  

## 相關倉庫
| 模型 | 路徑 |
| :----- | :----- |
Unet | https://github.com/bubbliiiing/unet-pytorch  
PSPnet | https://github.com/bubbliiiing/pspnet-pytorch
deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch

### 性能情況
**unet並不適合VOC此類數據集，其更適合特徵少，需要淺層特徵的醫藥數據集之類的。**
| 訓練數據集 | 權值文件名稱 | 測試數據集 | 輸入圖片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [unet_vgg_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth) | VOC-Val12 | 512x512| 58.78 | 
| VOC12+SBD | [unet_resnet_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth) | VOC-Val12 | 512x512| 67.53 | 

### 所需環境
torch==1.2.0    
torchvision==0.4.0   

### 文件下載
訓練所需的權值可在百度網盤中下載。    
鏈接: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A    
提取碼: 6n2c   

VOC拓展數據集的百度網盤如下：   
鏈接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng    
提取碼: 44mk   

### 訓練步驟
#### 一、訓練voc數據集
1、將我提供的voc數據集放入VOCdevkit中（無需運行voc_annotation.py）。  
2、運行train.py進行訓練，默認參數已經對應voc數據集所需要的參數了。  

#### 二、訓練自己的數據集
1、本文使用VOC格式進行訓練。  
2、訓練前將標簽文件放在VOCdevkit文件夾下的VOC2007文件夾下的SegmentationClass中。    
3、訓練前將圖片文件放在VOCdevkit文件夾下的VOC2007文件夾下的JPEGImages中。    
4、在訓練前利用voc_annotation.py文件生成對應的txt。    
5、注意修改train.py的num_classes為分類個數+1。    
6、運行train.py即可開始訓練。  

#### 三、訓練醫藥數據集
1、下載VGG的預訓練權重到model_data下麵。  
2、按照默認參數運行train_medical.py即可開始訓練。

### 預測步驟
#### 一、使用預訓練權重
##### a、VOC預訓練權重
1. 下載完庫後解壓，如果想要利用voc訓練好的權重進行預測，在百度網盤或者release下載權值，放入model_data，運行即可預測。  
```python
img/street.jpg
```    
2. 在predict.py裡面進行設置可以進行fps測試和video視頻檢測。    
##### b、醫藥預訓練權重
1. 下載完庫後解壓，如果想要利用醫藥數據集訓練好的權重進行預測，在百度網盤或者release下載權值，放入model_data，修改unet.py中的model_path和num_classes；
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夾下的權值文件
    #   訓練好後logs文件夾下存在多個權值文件，選擇驗證集損失較低的即可。
    #   驗證集損失較低不代表miou較高，僅代表該權值在驗證集上泛化性能較好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_medical.pth',
    #--------------------------------#
    #   所需要區分的類的個數+1
    #--------------------------------#
    "num_classes"   : 2,
    #--------------------------------#
    #   所使用的的主乾網路：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   輸入圖片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend參數用於控制是否
    #   讓識別結果和原圖混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #--------------------------------#
    "cuda"          : True,
}
```
2. 運行即可預測。  
```python
img/cell.png
```
#### 二、使用自己訓練的權重
1. 按照訓練步驟訓練。    
2. 在unet.py文件裡面，在如下部分修改model_path、backbone和num_classes使其對應訓練好的文件；**model_path對應logs文件夾下麵的權值文件**。    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夾下的權值文件
    #   訓練好後logs文件夾下存在多個權值文件，選擇驗證集損失較低的即可。
    #   驗證集損失較低不代表miou較高，僅代表該權值在驗證集上泛化性能較好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   所需要區分的類的個數+1
    #--------------------------------#
    "num_classes"   : 21,
    #--------------------------------#
    #   所使用的的主乾網路：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "vgg",
    #--------------------------------#
    #   輸入圖片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   blend參數用於控制是否
    #   讓識別結果和原圖混合
    #--------------------------------#
    "blend"         : True,
    #--------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #--------------------------------#
    "cuda"          : True,
}
```
3. 運行predict.py，輸入    
```python
img/street.jpg
```   
4. 在predict.py裡面進行設置可以進行fps測試和video視頻檢測。    

### 評估步驟
1、設置get_miou.py裡面的num_classes為預測的類的數量加1。  
2、設置get_miou.py裡面的name_classes為需要去區分的類別。  
3、運行get_miou.py即可獲得miou大小。  

## Reference
https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus