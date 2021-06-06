# AI-Audio-Recognition

### Load and save Audio data 
##### librosa

### Data augmentation

### Feature Extraction
##### mfcc, spectral_center, chroma, spectral_contrast

### Training Model
##### Simple CNN
##### GoogleNet


### Problems and tips
#### 資料處理
1. 切train, test set時考慮分類頻率、shuffle、用pickle存起來
2. 資料的維度問題
3. 寫Dataset的class

#### 模型
1. model的維度、maxpooling後的計算
2. train時，batch size設大一點(用server跑)一次可以看較多資料;validate大也沒差
3. output答案善用torch.argmax和F.softmax(x, dim=1)
4. 用picke存model時、可用model.state_dict()存參數
5. 存model記得創資料夾and改檔名


#### 訓練
1. 嘗試不同optimizor:adam, SGD, adabound
2. 檢查test上不同類別的表現、對較差的增加權重、擴增其資料

#### 環境
1. device 為cuda or cpu
2. mac資料夾中會出現'.DS_Store'檔案

