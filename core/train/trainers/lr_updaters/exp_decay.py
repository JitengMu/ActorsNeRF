
from configs import cfg

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def _update_lr(optimizer, iter_step):
    decay_rate = 0.1
    decay_steps = cfg.train.lrate_decay * 1000
    decay_value = decay_rate ** (iter_step / decay_steps)
    for param_group in optimizer.param_groups:
        if f"lr_{param_group['name']}" in cfg.train:
            base_lr = cfg.train[f"lr_{param_group['name']}"]
            new_lrate = base_lr * decay_value
        else:
            new_lrate = cfg.train.lr * decay_value
        param_group['lr'] = new_lrate

def update_lr(optimizer, iter_step):

    def get_customized_lr_names():
        return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]
    cus_lr_names = get_customized_lr_names()

    decay_rate = 0.1
    decay_steps = cfg.train.lrate_decay * 1000
    decay_value = decay_rate ** (iter_step / decay_steps)
    for param_group in optimizer.param_groups:
        flag = False
        for lr_name in cus_lr_names:
            if lr_name in f"lr_{param_group['name']}":
                base_lr = cfg.train[f"lr_{lr_name}"]
                new_lrate = base_lr * decay_value
                flag = True
        if flag==False:
            new_lrate = cfg.train.lr * decay_value
        param_group['lr'] = new_lrate