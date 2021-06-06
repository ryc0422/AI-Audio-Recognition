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
1. 切train, test set時考慮分類頻率、shuffle、用pickle存起來
2. 資料的維度問題
3. 寫Dataset的class
4. model的維度、maxpooling後的計算
5. output答案善用torch.argmax和F.softmax(x, dim=1)
6. device 為cuda or cpu
7. mac資料夾中會出現'.DS_Store'檔案
8. 用picke存model時、可用model.state_dict()存參數
