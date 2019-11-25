""" 
@ author: Qmh
@ file_name: build_net.py
@ time: 2019:11:20:10:04
""" 
from torch import nn
import torchvision.models as models
import models as customized_models
from args import args

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

def make_model(args):
    print("=> creating model '{}'".format(args.arch))
    # 加载预训练模型 
    model = models.__dict__[args.arch](progress=True)
    # model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # 最后一层全连接层
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, args.num_classes)
    )
    return model

if __name__=='__main__':
    all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    print(all_model)
    model = make_model(args)