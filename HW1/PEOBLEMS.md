問題匯總的博客地址為[https://blog.csdn.net/weixin_44791964/article/details/107517428](https://blog.csdn.net/weixin_44791964/article/details/107517428)。

# 問題匯總
## 1、下載問題
### a、代碼下載
**問：up主，可以給我發一份代碼嗎，代碼在哪里下載啊？ 
答：Github上的地址就在視頻簡介里。覆制一下就能進去下載了。**

**問：up主，為什麽我下載的代碼提示壓縮包損壞？
答：重新去Github下載。**

**問：up主，為什麽我下載的代碼和你在視頻以及博客上的代碼不一樣？
答：我常常會對代碼進行更新，最終以實際的代碼為準。**

### b、 權值下載
**問：up主，為什麽我下載的代碼里面，model_data下面沒有.pth或者.h5文件？ 
答：我一般會把權值上傳到Github和百度網盤，在GITHUB的README里面就能找到。**

### c、 數據集下載
**問：up主，XXXX數據集在哪里下載啊？
答：一般數據集的下載地址我會放在README里面，基本上都有，沒有的話請及時聯系我添加，直接發github的issue即可**。

## 2、環境配置問題
### a、現在庫中所用的環境
**pytorch代碼對應的pytorch版本為1.2，博客地址對應**[https://blog.csdn.net/weixin_44791964/article/details/106037141](https://blog.csdn.net/weixin_44791964/article/details/106037141)。

**keras代碼對應的tensorflow版本為1.13.2，keras版本是2.1.5，博客地址對應**[https://blog.csdn.net/weixin_44791964/article/details/104702142](https://blog.csdn.net/weixin_44791964/article/details/104702142)。

**tf2代碼對應的tensorflow版本為2.2.0，無需安裝keras，博客地址對應**[https://blog.csdn.net/weixin_44791964/article/details/109161493](https://blog.csdn.net/weixin_44791964/article/details/109161493)。

**問：你的代碼某某某版本的tensorflow和pytorch能用嘛？
答：最好按照我推薦的配置，配置教程也有！其它版本的我沒有試過！可能出現問題但是一般問題不大。僅需要改少量代碼即可。**

### b、30系列顯卡環境配置
30系顯卡由於框架更新不可使用上述環境配置教程。
當前我已經測試的可以用的30顯卡配置如下：
**pytorch代碼對應的pytorch版本為1.7.0，cuda為11.0，cudnn為8.0.5**。

**keras代碼無法在win10下配置cuda11，在ubuntu下可以百度查詢一下，配置tensorflow版本為1.15.4，keras版本是2.1.5或者2.3.1（少量函數接口不同，代碼可能還需要少量調整。）**

**tf2代碼對應的tensorflow版本為2.4.0，cuda為11.0，cudnn為8.0.5**。

### c、GPU利用問題與環境使用問題
**問：為什麽我安裝了tensorflow-gpu但是卻沒用利用GPU進行訓練呢？
答：確認tensorflow-gpu已經裝好，利用pip list查看tensorflow版本，然後查看任務管理器或者利用nvidia命令看看是否使用了gpu進行訓練，任務管理器的話要看顯存使用情況。**

**問：up主，我好像沒有在用gpu進行訓練啊，怎麽看是不是用了GPU進行訓練？
答：查看是否使用GPU進行訓練一般使用NVIDIA在命令行的查看命令，如果要看任務管理器的話，請看性能部分GPU的顯存是否利用，或者查看任務管理器的Cuda，而非Copy。**
![在這里插入圖片描述](https://img-blog.csdnimg.cn/20201013234241524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

**問：up主，為什麽我按照你的環境配置後還是不能使用？
答：請把你的GPU、CUDA、CUDNN、TF版本以及PYTORCH版本B站私聊告訴我。**

**問：出現如下錯誤**
```python
Traceback (most recent call last):
  File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
 from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
pywrap_tensorflow_internal = swig_import_helper()
  File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\imp.py", line 243, in load_modulereturn load_dynamic(name, filename, file)
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\imp.py", line 343, in load_dynamic
    return _load(spec)
ImportError: DLL load failed: 找不到指定的模塊。
```
**答：如果沒重啟過就重啟一下，否則重新按照步驟安裝，還無法解決則把你的GPU、CUDA、CUDNN、TF版本以及PYTORCH版本私聊告訴我。**

### d、no module問題
**問：為什麽提示說no module name utils.utils（no module name nets.yolo、no module name nets.ssd等一系列問題）啊？
答：utils並不需要用pip裝，它就在我上傳的倉庫的根目錄，出現這個問題的原因是根目錄不對，查查相對目錄和根目錄的概念。查了基本上就明白了。**

**問：為什麽提示說no module name matplotlib（no module name PIL，no module name cv2等等）？
答：這個庫沒安裝打開命令行安裝就好。pip install matplotlib**

**問：為什麽我已經用pip裝了opencv（pillow、matplotlib等），還是提示no module name cv2？
答：沒有激活環境裝，要激活對應的conda環境進行安裝才可以正常使用**

**問：為什麽提示說No module named 'torch' ？
答：其實我也真的很想知道為什麽會有這個問題……這個pytorch沒裝是什麽情況？一般就倆情況，一個是真的沒裝，還有一個是裝到其它環境了，當前激活的環境不是自己裝的環境。**

**問：為什麽提示說No module named 'tensorflow' ？
答：同上。**

### e、cuda安裝失敗問題
一般cuda安裝前需要安裝Visual Studio，裝個2017版本即可。

### f、Ubuntu系統問題
**所有代碼在Ubuntu下可以使用，我兩個系統都試過。**

### g、VSCODE提示錯誤的問題
**問：為什麽在VSCODE里面提示一大堆的錯誤啊？
答：我也提示一大堆的錯誤，但是不影響，是VSCODE的問題，如果不想看錯誤的話就裝Pycharm。**

### h、使用cpu進行訓練與預測的問題
**對於keras和tf2的代碼而言，如果想用cpu進行訓練和預測，直接裝cpu版本的tensorflow就可以了。**

**對於pytorch的代碼而言，如果想用cpu進行訓練和預測，需要將cuda=True修改成cuda=False。**

### i、tqdm沒有pos參數問題
**問：運行代碼提示'tqdm' object has no attribute 'pos'。
答：重裝tqdm，換個版本就可以了。**

### j、提示decode(“utf-8”)的問題
**由於h5py庫的更新，安裝過程中會自動安裝h5py=3.0.0以上的版本，會導致decode("utf-8")的錯誤！
各位一定要在安裝完tensorflow後利用命令裝h5py=2.10.0！**
```
pip install h5py==2.10.0
```

### k、提示TypeError: __array__() takes 1 positional argument but 2 were given錯誤
可以修改pillow版本解決。
```
pip install pillow==8.2.0
```

### l、其它問題
**問：為什麽提示TypeError: cat() got an unexpected keyword argument 'axis'，Traceback (most recent call last)，AttributeError: 'Tensor' object has no attribute 'bool'？
答：這是版本問題，建議使用torch1.2以上版本**
**其它有很多稀奇古怪的問題，很多是版本問題，建議按照我的視頻教程安裝Keras和tensorflow。比如裝的是tensorflow2，就不用問我說為什麽我沒法運行Keras-yolo啥的。那是必然不行的。**

## 3、目標檢測庫問題匯總（人臉檢測和分類庫也可參考）
### a、shape不匹配問題
#### 1）、訓練時shape不匹配問題
**問：up主，為什麽運行train.py會提示shape不匹配啊？
答：在keras環境中，因為你訓練的種類和原始的種類不同，網絡結構會變化，所以最尾部的shape會有少量不匹配。**

#### 2）、預測時shape不匹配問題
**問：為什麽我運行predict.py會提示我說shape不匹配呀。
在Pytorch里面是這樣的：**
![在這里插入圖片描述](https://img-blog.csdnimg.cn/20200722171631901.png)
在Keras里面是這樣的：
![在這里插入圖片描述](https://img-blog.csdnimg.cn/20200722171523380.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70)
**答：原因主要有仨：
1、在ssd、FasterRCNN里面，可能是train.py里面的num_classes沒改。
2、model_path沒改。
3、classes_path沒改。
請檢查清楚了！確定自己所用的model_path和classes_path是對應的！訓練的時候用到的num_classes或者classes_path也需要檢查！**

### b、顯存不足問題
**問：為什麽我運行train.py下面的命令行閃的賊快，還提示OOM啥的？ 
答：這是在keras中出現的，爆顯存了，可以改小batch_size，SSD的顯存占用率是最小的，建議用SSD；
2G顯存：SSD、YOLOV4-TINY
4G顯存：YOLOV3
6G顯存：YOLOV4、Retinanet、M2det、Efficientdet、Faster RCNN等
8G+顯存：隨便選吧。**
**需要注意的是，受到BatchNorm2d影響，batch_size不可為1，至少為2。**

**問：為什麽提示 RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; 15.07 GiB reserved in total by PyTorch)？ 
答：這是pytorch中出現的，爆顯存了，同上。**

**問：為什麽我顯存都沒利用，就直接爆顯存了？ 
答：都爆顯存了，自然就不利用了，模型沒有開始訓練。**
### c、訓練問題（凍結訓練，LOSS問題、訓練效果問題等）
**問：為什麽要凍結訓練和解凍訓練呀？
答：這是遷移學習的思想，因為神經網絡主幹特征提取部分所提取到的特征是通用的，我們凍結起來訓練可以加快訓練效率，也可以防止權值被破壞。**
在凍結階段，模型的主幹被凍結了，特征提取網絡不發生改變。占用的顯存較小，僅對網絡進行微調。
在解凍階段，模型的主幹不被凍結了，特征提取網絡會發生改變。占用的顯存較大，網絡所有的參數都會發生改變。

**問：為什麽我的網絡不收斂啊，LOSS是XXXX。
答：不同網絡的LOSS不同，LOSS只是一個參考指標，用於查看網絡是否收斂，而非評價網絡好壞，我的yolo代碼都沒有歸一化，所以LOSS值看起來比較高，LOSS的值不重要，重要的是是否在變小，預測是否有效果。**

**問：為什麽我的訓練效果不好？預測了沒有框（框不準）。
答：**

考慮幾個問題：
1、目標信息問題，查看2007_train.txt文件是否有目標信息，沒有的話請修改voc_annotation.py。
2、數據集問題，小於500的自行考慮增加數據集，同時測試不同的模型，確認數據集是好的。
3、是否解凍訓練，如果數據集分布與常規畫面差距過大需要進一步解凍訓練，調整主幹，加強特征提取能力。
4、網絡問題，比如SSD不適合小目標，因為先驗框固定了。
5、訓練時長問題，有些同學只訓練了幾代表示沒有效果，按默認參數訓練完。
6、確認自己是否按照步驟去做了，如果比如voc_annotation.py里面的classes是否修改了等。
7、不同網絡的LOSS不同，LOSS只是一個參考指標，用於查看網絡是否收斂，而非評價網絡好壞，LOSS的值不重要，重要的是是否收斂。

**問：我怎麽出現了gbk什麽的編碼錯誤啊：**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6 in position 446: illegal multibyte sequence
```
**答：標簽和路徑不要使用中文，如果一定要使用中文，請注意處理的時候編碼的問題，改成打開文件的encoding方式改為utf-8。**

**問：我的圖片是xxx*xxx的分辨率的，可以用嗎！**
**答：可以用，代碼里面會自動進行resize或者數據增強。**

**問：怎麽進行多GPU訓練？
答：pytorch的大多數代碼可以直接使用gpu訓練，keras的話直接百度就好了，實現並不覆雜，我沒有多卡沒法詳細測試，還需要各位同學自己努力了。**
### d、灰度圖問題
**問：能不能訓練灰度圖（預測灰度圖）啊？
答：我的大多數庫會將灰度圖轉化成RGB進行訓練和預測，如果遇到代碼不能訓練或者預測灰度圖的情況，可以嘗試一下在get_random_data里面將Image.open後的結果轉換成RGB，預測的時候也這樣試試。（僅供參考）**

### e、斷點續練問題
**問：我已經訓練過幾個世代了，能不能從這個基礎上繼續開始訓練
答：可以，你在訓練前，和載入預訓練權重一樣載入訓練過的權重就行了。一般訓練好的權重會保存在logs文件夾里面，將model_path修改成你要開始的權值的路徑即可。**

### f、預訓練權重的問題
**問：如果我要訓練其它的數據集，預訓練權重要怎麽辦啊？**
**答：數據的預訓練權重對不同數據集是通用的，因為特征是通用的，預訓練權重對於99%的情況都必須要用，不用的話權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好。**

**問：up，我修改了網絡，預訓練權重還能用嗎？
答：修改了主幹的話，如果不是用的現有的網絡，基本上預訓練權重是不能用的，要麽就自己判斷權值里卷積核的shape然後自己匹配，要麽只能自己預訓練去了；修改了後半部分的話，前半部分的主幹部分的預訓練權重還是可以用的，如果是pytorch代碼的話，需要自己修改一下載入權值的方式，判斷shape後載入，如果是keras代碼，直接by_name=True,skip_mismatch=True即可。**
權值匹配的方式可以參考如下：
```python
# 加快模型訓練的效率
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
    try:    
        if np.shape(model_dict[k]) ==  np.shape(v):
            a[k]=v
    except:
        pass
model_dict.update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**問：我要怎麽不使用預訓練權重啊？
答：把載入預訓練權重的代碼注釋了就行。**

**問：為什麽我不使用預訓練權重效果這麽差啊？
答：因為隨機初始化的權值不好，提取的特征不好，也就導致了模型訓練的效果不好，voc07+12、coco+voc07+12效果都不一樣，預訓練權重還是非常重要的。**

### g、視頻檢測問題與攝像頭檢測問題
**問：怎麽用攝像頭檢測呀？
答：predict.py修改參數可以進行攝像頭檢測，也有視頻詳細解釋了攝像頭檢測的思路。**

**問：怎麽用視頻檢測呀？
答：同上**
### h、從0開始訓練問題
**問：怎麽在模型上從0開始訓練？
答：在算力不足與調參能力不足的情況下從0開始訓練毫無意義。模型特征提取能力在隨機初始化參數的情況下非常差。沒有好的參數調節能力和算力，無法使得網絡正常收斂。**
如果一定要從0開始，那麽訓練的時候請注意幾點：
 - 不載入預訓練權重。 
 - 不要進行凍結訓練，注釋凍結模型的代碼。

**問：為什麽我不使用預訓練權重效果這麽差啊？
答：因為隨機初始化的權值不好，提取的特征不好，也就導致了模型訓練的效果不好，voc07+12、coco+voc07+12效果都不一樣，預訓練權重還是非常重要的。**

### i、保存問題
**問：檢測完的圖片怎麽保存？
答：一般目標檢測用的是Image，所以查詢一下PIL庫的Image如何進行保存。詳細看看predict.py文件的注釋。**

**問：怎麽用視頻保存呀？
答：詳細看看predict.py文件的注釋。**

### j、遍歷問題
**問：如何對一個文件夾的圖片進行遍歷？
答：一般使用os.listdir先找出文件夾里面的所有圖片，然後根據predict.py文件里面的執行思路檢測圖片就行了，詳細看看predict.py文件的注釋。**

**問：如何對一個文件夾的圖片進行遍歷？並且保存。
答：遍歷的話一般使用os.listdir先找出文件夾里面的所有圖片，然後根據predict.py文件里面的執行思路檢測圖片就行了。保存的話一般目標檢測用的是Image，所以查詢一下PIL庫的Image如何進行保存。如果有些庫用的是cv2，那就是查一下cv2怎麽保存圖片。詳細看看predict.py文件的注釋。**

### k、路徑問題（No such file or directory）
**問：我怎麽出現了這樣的錯誤呀：**
```python
FileNotFoundError: 【Errno 2】 No such file or directory
……………………………………
……………………………………
```
**答：去檢查一下文件夾路徑，查看是否有對應文件；並且檢查一下2007_train.txt，其中文件路徑是否有錯。**
關於路徑有幾個重要的點：
**文件夾名稱中一定不要有空格。
注意相對路徑和絕對路徑。
多百度路徑相關的知識。**

**所有的路徑問題基本上都是根目錄問題，好好查一下相對目錄的概念！**
### l、和原版比較問題
**問：你這個代碼和原版比怎麽樣，可以達到原版的效果麽？
答：基本上可以達到，我都用voc數據測過，我沒有好顯卡，沒有能力在coco上測試與訓練。**

**問：你有沒有實現yolov4所有的tricks，和原版差距多少？
答：並沒有實現全部的改進部分，由於YOLOV4使用的改進實在太多了，很難完全實現與列出來，這里只列出來了一些我比較感興趣，而且非常有效的改進。論文中提到的SAM（注意力機制模塊），作者自己的源碼也沒有使用。還有其它很多的tricks，不是所有的tricks都有提升，我也沒法實現全部的tricks。至於和原版的比較，我沒有能力訓練coco數據集，根據使用過的同學反應差距不大。**

### m、FPS問題（檢測速度問題）
**問：你這個FPS可以到達多少，可以到 XX FPS麽？
答：FPS和機子的配置有關，配置高就快，配置低就慢。**

**問：為什麽我用服務器去測試yolov4（or others）的FPS只有十幾？
答：檢查是否正確安裝了tensorflow-gpu或者pytorch的gpu版本，如果已經正確安裝，可以去利用time.time()的方法查看detect_image里面，哪一段代碼耗時更長（不僅只有網絡耗時長，其它處理部分也會耗時，如繪圖等）。**

**問：為什麽論文中說速度可以達到XX，但是這里卻沒有？
答：檢查是否正確安裝了tensorflow-gpu或者pytorch的gpu版本，如果已經正確安裝，可以去利用time.time()的方法查看detect_image里面，哪一段代碼耗時更長（不僅只有網絡耗時長，其它處理部分也會耗時，如繪圖等）。有些論文還會使用多batch進行預測，我並沒有去實現這個部分。**

### n、預測圖片不顯示問題
**問：為什麽你的代碼在預測完成後不顯示圖片？只是在命令行告訴我有什麽目標。
答：給系統安裝一個圖片查看器就行了。**

### o、算法評價問題（目標檢測的map、PR曲線、Recall、Precision等）
**問：怎麽計算map？
答：看map視頻，都一個流程。**

**問：計算map的時候，get_map.py里面有一個MINOVERLAP是什麽用的，是iou嗎？
答：是iou，它的作用是判斷預測框和真實框的重合成度，如果重合程度大於MINOVERLAP，則預測正確。**

**問：為什麽get_map.py里面的self.confidence（self.score）要設置的那麽小？
答：看一下map的視頻的原理部分，要知道所有的結果然後再進行pr曲線的繪制。**

**問：能不能說說怎麽繪制PR曲線啥的呀。
答：可以看mAP視頻，結果里面有PR曲線。**

**問：怎麽計算Recall、Precision指標。
答：這倆指標應該是相對於特定的置信度的，計算map的時候也會獲得。**

### p、coco數據集訓練問題
**問：目標檢測怎麽訓練COCO數據集啊？。
答：coco數據訓練所需要的txt文件可以參考qqwweee的yolo3的庫，格式都是一樣的。**

### q、模型優化（模型修改）問題
**問：up，YOLO系列使用Focal LOSS的代碼你有嗎，有提升嗎？
答：很多人試過，提升效果也不大（甚至變的更Low），它自己有自己的正負樣本的平衡方式。**

**問：up，我修改了網絡，預訓練權重還能用嗎？
答：修改了主幹的話，如果不是用的現有的網絡，基本上預訓練權重是不能用的，要麽就自己判斷權值里卷積核的shape然後自己匹配，要麽只能自己預訓練去了；修改了後半部分的話，前半部分的主幹部分的預訓練權重還是可以用的，如果是pytorch代碼的話，需要自己修改一下載入權值的方式，判斷shape後載入，如果是keras代碼，直接by_name=True,skip_mismatch=True即可。**
權值匹配的方式可以參考如下：
```python
# 加快模型訓練的效率
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
    try:    
        if np.shape(model_dict[k]) ==  np.shape(v):
            a[k]=v
    except:
        pass
model_dict.update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**問：up，怎麽修改模型啊，我想發個小論文！
答：建議看看yolov3和yolov4的區別，然後看看yolov4的論文，作為一個大型調參現場非常有參考意義，使用了很多tricks。我能給的建議就是多看一些經典模型，然後拆解里面的亮點結構並使用。**

### r、部署問題
我沒有具體部署到手機等設備上過，所以很多部署問題我並不了解……

## 4、語義分割庫問題匯總
### a、shape不匹配問題
#### 1）、訓練時shape不匹配問題
**問：up主，為什麽運行train.py會提示shape不匹配啊？
答：在keras環境中，因為你訓練的種類和原始的種類不同，網絡結構會變化，所以最尾部的shape會有少量不匹配。**

#### 2）、預測時shape不匹配問題
**問：為什麽我運行predict.py會提示我說shape不匹配呀。
在Pytorch里面是這樣的：**
![在這里插入圖片描述](https://img-blog.csdnimg.cn/20200722171631901.png)
在Keras里面是這樣的：
![在這里插入圖片描述](https://img-blog.csdnimg.cn/20200722171523380.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70)
**答：原因主要有二：
1、train.py里面的num_classes沒改。
2、預測時num_classes沒改。
請檢查清楚！訓練和預測的時候用到的num_classes都需要檢查！**

### b、顯存不足問題
**問：為什麽我運行train.py下面的命令行閃的賊快，還提示OOM啥的？ 
答：這是在keras中出現的，爆顯存了，可以改小batch_size。**

**需要注意的是，受到BatchNorm2d影響，batch_size不可為1，至少為2。**

**問：為什麽提示 RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; 15.07 GiB reserved in total by PyTorch)？ 
答：這是pytorch中出現的，爆顯存了，同上。**

**問：為什麽我顯存都沒利用，就直接爆顯存了？ 
答：都爆顯存了，自然就不利用了，模型沒有開始訓練。**

### c、訓練問題（凍結訓練，LOSS問題、訓練效果問題等）
**問：為什麽要凍結訓練和解凍訓練呀？
答：這是遷移學習的思想，因為神經網絡主幹特征提取部分所提取到的特征是通用的，我們凍結起來訓練可以加快訓練效率，也可以防止權值被破壞。**
**在凍結階段，模型的主幹被凍結了，特征提取網絡不發生改變。占用的顯存較小，僅對網絡進行微調。**
**在解凍階段，模型的主幹不被凍結了，特征提取網絡會發生改變。占用的顯存較大，網絡所有的參數都會發生改變。**

**問：為什麽我的網絡不收斂啊，LOSS是XXXX。
答：不同網絡的LOSS不同，LOSS只是一個參考指標，用於查看網絡是否收斂，而非評價網絡好壞，我的yolo代碼都沒有歸一化，所以LOSS值看起來比較高，LOSS的值不重要，重要的是是否在變小，預測是否有效果。**

**問：為什麽我的訓練效果不好？預測了沒有目標，結果是一片黑。
答：**
**考慮幾個問題：
1、數據集問題，這是最重要的問題。小於500的自行考慮增加數據集；一定要檢查數據集的標簽，視頻中詳細解析了VOC數據集的格式，但並不是有輸入圖片有輸出標簽即可，還需要確認標簽的每一個像素值是否為它對應的種類。很多同學的標簽格式不對，最常見的錯誤格式就是標簽的背景為黑，目標為白，此時目標的像素點值為255，無法正常訓練，目標需要為1才行。
2、是否解凍訓練，如果數據集分布與常規畫面差距過大需要進一步解凍訓練，調整主幹，加強特征提取能力。
3、網絡問題，可以嘗試不同的網絡。
4、訓練時長問題，有些同學只訓練了幾代表示沒有效果，按默認參數訓練完。
5、確認自己是否按照步驟去做了。
6、不同網絡的LOSS不同，LOSS只是一個參考指標，用於查看網絡是否收斂，而非評價網絡好壞，LOSS的值不重要，重要的是是否收斂。**



**問：為什麽我的訓練效果不好？對小目標預測不準確。
答：對於deeplab和pspnet而言，可以修改一下downsample_factor，當downsample_factor為16的時候下采樣倍數過多，效果不太好，可以修改為8。**

**問：我怎麽出現了gbk什麽的編碼錯誤啊：**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6 in position 446: illegal multibyte sequence
```
**答：標簽和路徑不要使用中文，如果一定要使用中文，請注意處理的時候編碼的問題，改成打開文件的encoding方式改為utf-8。**

**問：我的圖片是xxx*xxx的分辨率的，可以用嗎！**
**答：可以用，代碼里面會自動進行resize或者數據增強。**

**問：怎麽進行多GPU訓練？
答：pytorch的大多數代碼可以直接使用gpu訓練，keras的話直接百度就好了，實現並不覆雜，我沒有多卡沒法詳細測試，還需要各位同學自己努力了。**

### d、灰度圖問題
**問：能不能訓練灰度圖（預測灰度圖）啊？
答：我的大多數庫會將灰度圖轉化成RGB進行訓練和預測，如果遇到代碼不能訓練或者預測灰度圖的情況，可以嘗試一下在get_random_data里面將Image.open後的結果轉換成RGB，預測的時候也這樣試試。（僅供參考）**

### e、斷點續練問題
**問：我已經訓練過幾個世代了，能不能從這個基礎上繼續開始訓練
答：可以，你在訓練前，和載入預訓練權重一樣載入訓練過的權重就行了。一般訓練好的權重會保存在logs文件夾里面，將model_path修改成你要開始的權值的路徑即可。**

### f、預訓練權重的問題

**問：如果我要訓練其它的數據集，預訓練權重要怎麽辦啊？**
**答：數據的預訓練權重對不同數據集是通用的，因為特征是通用的，預訓練權重對於99%的情況都必須要用，不用的話權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好。**

**問：up，我修改了網絡，預訓練權重還能用嗎？
答：修改了主幹的話，如果不是用的現有的網絡，基本上預訓練權重是不能用的，要麽就自己判斷權值里卷積核的shape然後自己匹配，要麽只能自己預訓練去了；修改了後半部分的話，前半部分的主幹部分的預訓練權重還是可以用的，如果是pytorch代碼的話，需要自己修改一下載入權值的方式，判斷shape後載入，如果是keras代碼，直接by_name=True,skip_mismatch=True即可。**
權值匹配的方式可以參考如下：

```python
# 加快模型訓練的效率
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
    try:    
        if np.shape(model_dict[k]) ==  np.shape(v):
            a[k]=v
    except:
        pass
model_dict.update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**問：我要怎麽不使用預訓練權重啊？
答：把載入預訓練權重的代碼注釋了就行。**

**問：為什麽我不使用預訓練權重效果這麽差啊？
答：因為隨機初始化的權值不好，提取的特征不好，也就導致了模型訓練的效果不好，預訓練權重還是非常重要的。**

### g、視頻檢測問題與攝像頭檢測問題
**問：怎麽用攝像頭檢測呀？
答：predict.py修改參數可以進行攝像頭檢測，也有視頻詳細解釋了攝像頭檢測的思路。**

**問：怎麽用視頻檢測呀？
答：同上**

### h、從0開始訓練問題
**問：怎麽在模型上從0開始訓練？
答：在算力不足與調參能力不足的情況下從0開始訓練毫無意義。模型特征提取能力在隨機初始化參數的情況下非常差。沒有好的參數調節能力和算力，無法使得網絡正常收斂。**
如果一定要從0開始，那麽訓練的時候請注意幾點：
 - 不載入預訓練權重。
 - 不要進行凍結訓練，注釋凍結模型的代碼。

**問：為什麽我不使用預訓練權重效果這麽差啊？
答：因為隨機初始化的權值不好，提取的特征不好，也就導致了模型訓練的效果不好，預訓練權重還是非常重要的。**

### i、保存問題
**問：檢測完的圖片怎麽保存？
答：一般目標檢測用的是Image，所以查詢一下PIL庫的Image如何進行保存。詳細看看predict.py文件的注釋。**

**問：怎麽用視頻保存呀？
答：詳細看看predict.py文件的注釋。**

### j、遍歷問題
**問：如何對一個文件夾的圖片進行遍歷？
答：一般使用os.listdir先找出文件夾里面的所有圖片，然後根據predict.py文件里面的執行思路檢測圖片就行了，詳細看看predict.py文件的注釋。**

**問：如何對一個文件夾的圖片進行遍歷？並且保存。
答：遍歷的話一般使用os.listdir先找出文件夾里面的所有圖片，然後根據predict.py文件里面的執行思路檢測圖片就行了。保存的話一般目標檢測用的是Image，所以查詢一下PIL庫的Image如何進行保存。如果有些庫用的是cv2，那就是查一下cv2怎麽保存圖片。詳細看看predict.py文件的注釋。**

### k、路徑問題（No such file or directory）
**問：我怎麽出現了這樣的錯誤呀：**
```python
FileNotFoundError: 【Errno 2】 No such file or directory
……………………………………
……………………………………
```

**答：去檢查一下文件夾路徑，查看是否有對應文件；並且檢查一下2007_train.txt，其中文件路徑是否有錯。**
關於路徑有幾個重要的點：
**文件夾名稱中一定不要有空格。
注意相對路徑和絕對路徑。
多百度路徑相關的知識。**

**所有的路徑問題基本上都是根目錄問題，好好查一下相對目錄的概念！**

### l、FPS問題（檢測速度問題）
**問：你這個FPS可以到達多少，可以到 XX FPS麽？
答：FPS和機子的配置有關，配置高就快，配置低就慢。**

**問：為什麽論文中說速度可以達到XX，但是這里卻沒有？
答：檢查是否正確安裝了tensorflow-gpu或者pytorch的gpu版本，如果已經正確安裝，可以去利用time.time()的方法查看detect_image里面，哪一段代碼耗時更長（不僅只有網絡耗時長，其它處理部分也會耗時，如繪圖等）。有些論文還會使用多batch進行預測，我並沒有去實現這個部分。**

### m、預測圖片不顯示問題
**問：為什麽你的代碼在預測完成後不顯示圖片？只是在命令行告訴我有什麽目標。
答：給系統安裝一個圖片查看器就行了。**

### n、算法評價問題（miou）
**問：怎麽計算miou？
答：參考視頻里的miou測量部分。**

**問：怎麽計算Recall、Precision指標。
答：現有的代碼還無法獲得，需要各位同學理解一下混淆矩陣的概念，然後自行計算一下。**

### o、模型優化（模型修改）問題
**問：up，我修改了網絡，預訓練權重還能用嗎？
答：修改了主幹的話，如果不是用的現有的網絡，基本上預訓練權重是不能用的，要麽就自己判斷權值里卷積核的shape然後自己匹配，要麽只能自己預訓練去了；修改了後半部分的話，前半部分的主幹部分的預訓練權重還是可以用的，如果是pytorch代碼的話，需要自己修改一下載入權值的方式，判斷shape後載入，如果是keras代碼，直接by_name=True,skip_mismatch=True即可。**
權值匹配的方式可以參考如下：

```python
# 加快模型訓練的效率
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
    try:    
        if np.shape(model_dict[k]) ==  np.shape(v):
            a[k]=v
    except:
        pass
model_dict.update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**問：up，怎麽修改模型啊，我想發個小論文！
答：建議看看目標檢測中yolov4的論文，作為一個大型調參現場非常有參考意義，使用了很多tricks。我能給的建議就是多看一些經典模型，然後拆解里面的亮點結構並使用。常用的tricks如注意力機制什麽的，可以試試。**

### p、部署問題
我沒有具體部署到手機等設備上過，所以很多部署問題我並不了解……

## 5、交流群問題
**問：up，有沒有QQ群啥的呢？
答：沒有沒有，我沒有時間管理QQ群……**

## 6、怎麽學習的問題
**問：up，你的學習路線怎麽樣的？我是個小白我要怎麽學？
答：這里有幾點需要注意哈
1、我不是高手，很多東西我也不會，我的學習路線也不一定適用所有人。
2、我實驗室不做深度學習，所以我很多東西都是自學，自己摸索，正確與否我也不知道。
3、我個人覺得學習更靠自學**
學習路線的話，我是先學習了莫煩的python教程，從tensorflow、keras、pytorch入門，入門完之後學的SSD，YOLO，然後了解了很多經典的卷積網，後面就開始學很多不同的代碼了，我的學習方法就是一行一行的看，了解整個代碼的執行流程，特征層的shape變化等，花了很多時間也沒有什麽捷徑，就是要花時間吧。