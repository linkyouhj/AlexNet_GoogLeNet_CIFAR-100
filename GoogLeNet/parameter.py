import torch
from model import GoogLeNet
from torchsummary import summary
import torch.nn as nn
# 모델 정의
# model = GoogLeNet(num_classes=100).to('cpu')
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True,aux_logits=True)

# 저장된 모델 로드 (CPU로 로드)
# model.load_state_dict(torch.load('./googlenet_cifar100.ckpt', map_location=torch.device('cpu')))
# model.pre_layer[0] = torch.nn.Conv2d(3, 64, kernel_size=7, padding=2, stride = 2, bias=False)
# model.pre_layer[3] = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride = 2, bias=False)
# model.pre_layer = nn.Sequential(
#     model.pre_layer[0:3],
#     model.pre_layer[6:],  # 두 번째 nn.Conv2d 제거
# )
# 총 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
print("총 파라미터 수:", total_params)
x=torch.rand(3,3,224,224)
print(summary(model,input_size=(3,32,32)))