# NIA Tool Segmentation  
**NIA Too Segmentation (Sep 16, 2020  ~ Feb 28,2021)**  


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
$ python3 evaluate_single.py --num_image {NUM}
```  
