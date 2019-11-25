# skin_disease_two_classification
### Resnet50进行迁移学习实现图片二分类

#### 内容简介
本文使用预训练的Resnet50网络对皮肤病图片进行二分类，基于portch框架。
- [本文代码](https://github.com/qmh1234567/skin_disease_two_classification)
- 参考了一个图片多分类的项目:[项目地址](https://github.com/ikkyu-wen/huawei-garbage)。


#### 数据集说明
数据集存放目录为： used_dataset , 共200张图片，标签为：benign(良性）、malignant(患病)。

- 数据集划分如下：
 
数据集类型    | benign | malignant | totall
-------- | ---- | ------- | -----
train  |  64 | 64 | 128 
val  |  16 |  16 |  32
test  |  20 | 20 | 40

#### 代码目录介绍
1. args.py  存放训练和测试所用的各种参数。
	  --mode字段表示运行模式：train or test. 
	  --model_path字段是训练模型的保存路径。
	  其余字段都有默认值。
2. create_dataset.py 该脚本是用来读json中的数据的，可以忽略。
3. data_gen.py 该脚本实现划分数据集以及数据增强和数据加载。
4. main.py 包含训练、评估和测试。
5. transform.py 实现图片增强。
6. utils.py 存放一些工具函数。
7. models/Res.py 是重写的ResNet各种类型的网络。
8. checkpoints 保存模型

#### 运行命令

```python
# 训练模型
python main.py --mode=train
# 测试模型
python main.py --mode=test --model_path='训练好的模型文件路径'   
```

#### main.py 脚本介绍
###### main()函数 实现模型的训练和评估
 -  step1: 加载数据 
```python
# data
    transformations = get_transforms(input_size=args.image_size,test_size=args.image_size)
    train_set = data_gen.Dataset(root=args.train_txt_path,transform=transformations['val_train'])
    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

    val_set = data_gen.ValDataset(root=args.val_txt_path,transform=transformations['val_test'])
    val_loader = data.DataLoader(val_set,batch_size=args.batch_size,shuffle=False)
```
   -  step2: 构建模型

```python
model = make_model(args)
    if use_cuda:
        model.cuda()
    
    # define loss function and optimizer
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
	
    optimizer = get_optimizer(model,args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
```
- step3: 模型的训练和评估
   
```python
# train
    for epoch in range(start_epoch,args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss,train_acc = train(train_loader,model,criterion,optimizer,epoch,use_cuda)
        test_loss,val_acc = val(val_loader,model,criterion,epoch,use_cuda)

        scheduler.step(test_loss)

        print(f'train_loss:{train_loss}\t val_loss:{test_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')

        # save_model
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc':train_acc,
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, single=True, checkpoint=args.checkpoint)

    print("best acc = ",best_acc)

```

###### train()函数 每个epoch下的模型训练过程
- 主要实现每个批次下梯度的反向传播，计算accuarcy 和 loss, 并更新，最后返回其均值。

```python
def train(train_loader,model,criterion,optimizer,epoch,use_cuda):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for (inputs,targets) in tqdm(train_loader):
        if use_cuda:
            inputs,targets = inputs.cuda(),targets.cuda(async=True)
        inputs,targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 梯度参数设为0
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy(outputs.data,targets.data)
        # inputs.size(0)=32
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(),inputs.size(0))

    return losses.avg,train_acc.avg
```
###### val()函数 每个epoch下的模型评估过程
- 主要代码与train()函数一致，但没有梯度的计算，还有将model.train()改成model.eval()。

```python
def val(val_loader,model,criterion,epoch,use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval() # 将模型设置为验证模式
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter(args.num_classes)
    for _,(inputs,targets) in enumerate(val_loader):
        if use_cuda:
            inputs,targets = inputs.cuda(),targets.cuda()
        inputs,targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        confusion_matrix.add(outputs.data.squeeze(),targets.long())
        acc1 = accuracy(outputs.data,targets.data)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        val_acc.update(acc1.item(),inputs.size(0))
    return losses.avg,val_acc.avg
```

###### test()函数 模型的测试

```python
def test(use_cuda):
    # data
    transformations = get_transforms(input_size=args.image_size,test_size=args.image_size)
    test_set = data_gen.TestDataset(root=args.test_txt_path,transform= transformations['test'])
    test_loader = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
    # load model
    model = make_model(args)
    
    if args.model_path:
        # 加载模型
        model.load_state_dict(torch.load(args.model_path))

    if use_cuda:
        model.cuda()

    # evaluate
    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval() # 设置成eval模式
        for (inputs,targets,paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs,targets = inputs.cuda(),targets.cuda()
            inputs,targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)  # (16,2)
            # dim=1 表示按行计算 即对每一行进行softmax
            # probability = torch.nn.functional.softmax(outputs,dim=1)[:,1].tolist()
            # probability = [1 if prob >= 0.5 else 0 for prob in probability]
            # 返回最大值的索引
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            y_pred.extend(probability)
        print("y_pred=",y_pred)

        accuracy = metrics.accuracy_score(y_true,y_pred)
        print("accuracy=",accuracy)
        confusion_matrix = metrics.confusion_matrix(y_true,y_pred)
        print("confusion_matrix=",confusion_matrix)
        print(metrics.classification_report(y_true,y_pred))
        # fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
        print("roc-auc score=",metrics.roc_auc_score(y_true,y_pred))
    
        res_dict = {
            'img_path':img_paths,
            'label':y_true,
            'predict':y_pred,

        }
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv,index=False)
        print(f"write to {args.result_csv} succeed ")
```
#### 实验结果
![result](https://img-blog.csdnimg.cn/20191124183216425.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4MjI4NjA1,size_16,color_FFFFFF,t_70)
