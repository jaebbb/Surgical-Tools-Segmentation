# Surgical Tool Segmentation  

## Install  
### Requirements  
```bash
$ pip install -r requirements.txt  
$ git clone https://www.github.com/nvidia/apex  
$ cd apex  
$ python setup.py install  
```  

### Preparing dataset  
- you can prepare datasets in [here](dataset/README.md)  

---

## Training    
```bash
$ python3 train.py --gpu-ids ${GPU_NUMBERS} 
```  

## Inference  
```bash
$ python3 evaluate_single.py --num_image {NUM} #segmentation 수행 결과 이미지
$ python3 evaluate_single.py --result matrix # 전체 test이미지에 대한 MIoU
```  
---
## Result   
<img src = https://user-images.githubusercontent.com/52495256/106235450-079a9e00-623e-11eb-95f0-eeafa3134bc5.png width='80%' hight='80%'>
<img src = https://user-images.githubusercontent.com/52495256/106229619-46762700-6231-11eb-8e6e-d69b0b2b52c1.png width='80%' hight='80%'>  


## Discussion  
19_mtclip의 경우 데이터 출현 빈도는 최하위는 아니였지만 전체적인 영역의 비율이  
0.12%밖에 되지 않아 학습을 하지 못하고 다수 background인 장기로 인식되었습니다.  

<img src = https://user-images.githubusercontent.com/52495256/106232823-4cbbd180-6238-11eb-9d64-fa0620e4cdba.png width='80%' hight='80%'>