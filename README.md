# TextCNN-For-Relation-Classification
### Introduction
An implementation for Relation Classification using textCNN 

Most part of code refers to  [roomylee](https://github.com/roomylee/cnn-relation-extraction),a little modification have been made to adapt current version of tensorflow.

The model is descriped as below in Paper [Relation Extraction: Perspective from Convolutional Neural Networks](https://cs.nyu.edu/~thien/pubs/vector15.pdf)
![image](https://github.com/DengChan/TextCNN-For-Relation-Classification/raw/master/images/model1.png)

To make the actual algorithm clear, I would like to present you this picture:

![image](https://github.com/DengChan/TextCNN-For-Relation-Classification/raw/master/images/model2.png)



### Usage
#### Requirements
* Tensorflow-1.11
* scikit-learn

#### File Organize
┃━ SemEval2010_task8_all_data  
&emsp;&emsp;┃━ SemEval2010_task8_training  
&emsp;&emsp;&emsp;&emsp;┃━ TRAIN_FILE.TXT  
&emsp;&emsp;&emsp;&emsp;┃━ TRAIN_TEST_DISTRIB.TXT  
┃━  runs  
&emsp;&emsp;┃━ 1547716564(timestap)  
&emsp;&emsp;&emsp;&emsp;┃━ checkpoints  
&emsp;&emsp;&emsp;&emsp;┃━ summaries  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;┃━ summaries  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;┃━ train  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;┃━ dev  
┃━  configure.py  
┃━  data_helpers.py  
┃━  model.py  
┃━  configure.py  
┃━  train.py  
┃━  utils.py  
┃━  GoogleNews-vectors-negative300.bin  

#### Train
* The train data is located in "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT".
* ["GoogleNews-vectors-negative300"](https://code.google.com/archive/p/word2vec/) is used as pre-trained word embeddings.Or you can get it at [BaiduYun](https://pan.baidu.com/s/1XuzfjBYIdye_UXytpv9qug]) , 2muq
* The parameters setting is located in configure.py.You can get help by command 
` python train --help `
* To run :
`python train (--[parameter_name] [value])`

#### Evaluation
* Test data is located in "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT".
* Parameter **checkpoint_dir** and the vacab_to_index,pos_to_index json file(generated by train step) paths **text_tokenizer_path**、**pos_tokenizer_path** must be given.Just like below:

```
python eval.py --checkpoint_dir "runs/1523902663/checkpoints/" --text_tokenizer_path "runs/1523902663/checkpoints/text_tokenizer.json --pos_tokenizer_path "runs/1523902663/checkpoints/pos_tokenizer.json"
```

#### Result
* Use macro average F1 score and Accuracy to evaluate the performance.
* I got a F1 score of 74.85% and the accuracy is 76% by using default settings
* Used dropout and l2 regularization , but overfitting is still a big problem .
* The best performance may reach 82% for F1, you can adjust the parameters to get better performance .