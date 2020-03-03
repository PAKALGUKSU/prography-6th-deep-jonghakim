# VGG16 with Skip Connection
VGG16에 skip connection을 구현한 모델입니다.

VGG16-codes.ipynb가 원래 작업하던 IPython Notebook, VGG.py가 해당 노트북을 python 파일로 변환한 파일입니다.

python test.py를 통해 정확도를 평가할 수 있는데, local cpu를 이용해 평가 코드를 실행하는 시간이 꽤 오래 걸려 10000개의 test data중 1000개만 평가하도록 설정했습니다.
10000개 전체 데이터는 colab 환경에서 test 진행하였으며, 전체 98%의 accuracy를 얻었습니다.

모델의 학습 결과 변수를 저장한 model.pth 파일은 https://colab.research.google.com/drive/11L-OZmfdZ_tIn0pTxHb3ApLoIAJCaFoI 에서 받을 수 있습니다.
해당 파일을 repository root directory에 다운로드받으면 됩니다.

# About Model
상술한대로 10000개의 test set으로 테스트를 진행한 결과 98%의 정확도를 얻을 수 있었습니다.

모델은 PyTorch를 통해 구현했으며, PyTorch의 실제 VGG-16 구현 코드를 참고했습니다.

Redisual connection은 첫 MaxPooling 결과 layer를 output channel 512, kernel size 2, stride 20, padding 5의 2*2 ConV layer를 통과하게 한 이후
해당 결과를 flatten해 기존 Fully Connected layer의 첫 layer input과 합해주는 방식으로 구현했습니다.

Loss는 CrossEntropy를, optimizer는 Adam을 사용했습니다(lr = 0.001)
