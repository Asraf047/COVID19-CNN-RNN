## Diagnosis of COVID-19 from X-rays Using Combined CNN-RNN Architecture with Transfer Learning
The confrontation of COVID-19 pandemic has become one of the promising challenges of the world healthcare. Accurate and fast diagnosis of COVID-19 cases is essential for correct medical treatment to control this pandemic. Compared with the reverse-transcription polymerase chain reaction (RT-PCR) method, chest radiography imaging techniques are shown to be more effective to detect coronavirus. For the limitation of available medical images, transfer learning is better suited to classify patterns in medical images. This paper presents a combined architecture of convolutional neural network (CNN) and recurrent neural network (RNN) to diagnose COVID-19 from chest X-rays. The deep transfer techniques used in this experiment are VGG19, DenseNet121, InceptionV3, and Inception-ResNetV2. CNN is used to extract complex features from samples and classified them using RNN. The VGG19-RNN architecture achieved the best performance among all the networks in terms of accuracy in our experiments. Finally, Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize class-specific regions of images that are responsible to make decision. The system achieved promising results compared to other existing systems and might be validated in the future when more samples would be available. The experiment demonstrated a good alternative method to diagnose COVID-19 for medical staff. 

## Overview
#### The overall system architecture:
![Picture2 - Copy](https://user-images.githubusercontent.com/31788789/111881337-2dd8f080-89da-11eb-9b59-bd2929bb677c.jpg)

#### The combined CNN-RNN architecture:
![22](https://user-images.githubusercontent.com/31788789/111881647-a8eed680-89db-11eb-8330-85bea9232bfc.jpg)

## Dataset
The dataset can be found in [here](https://www.kaggle.com/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset). For dataset related queries, please drop an email to amanullahoasraf@gmail.com

## Citation
Please cite our paper if you find the work useful:
```
@article{islam2020diagnosis,
  title={Diagnosis of COVID-19 from X-rays using combined CNN-RNN architecture with transfer learning},
  author={Islam, Md Milon and Islam, Md Zabirul and Asraf, Amanullah and Ding, Weiping},
  doi={10.1101/2020.08.24.20181339},
  year={2020},
  publisher={medRxiv}
}
```
```
M. M. Islam, M. Z. Islam, A. Asraf and W. Ding, "Diagnosis of COVID-19 from X-rays using combined CNN-RNN architecture with transfer learning," MedRxiv, Aug. 2020. [Online]. Available: https://www.medrxiv.org/content/10.1101/2020.08.24.20181339v1

```
> Paper: https://doi.org/10.1101/2020.08.24.20181339
