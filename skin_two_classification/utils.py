import torch
import adabound
import os

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, args):
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': args.lr * args.lr_fc_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})
    # parameters = model.parameters()
    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters,
                            # model.parameters(),
                               args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters,
                                # model.parameters(),
                                   args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters,
                                # model.parameters(),
                                args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    # elif args.optimizer == 'radam':
    #     return RAdam(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay)

    else:
        raise NotImplementedError



def save_checkpoint(state, is_best, single=True, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best:
        model_name = 'model_' + str(state['epoch']) + '_' + str(int(round(state['train_acc']*100, 0))) + '_' + str(int(round(state['acc']*100, 0))) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)



def accuracy(output, target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    # top1 accuracy
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True) # 返回最大的k个结果（按最大到小排序）
    
    pred = pred.t()  # 转置
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / batch_size)
    return res