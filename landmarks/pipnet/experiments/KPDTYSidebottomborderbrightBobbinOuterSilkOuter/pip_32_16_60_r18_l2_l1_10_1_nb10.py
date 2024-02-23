
class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 4
        self.init_lr = 0.0001
        self.num_epochs = 60000
        self.decay_steps = [30, 50]
        self.input_size = [256,256]   #[W H]
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 80
        self.save_interval = 5
        self.num_nb = 10
        self.use_gpu = True
        self.gpu_id = 0
        self.test_weight='epoch5660.pth'                #test_weight
        self.crop_size=[39,399,2551,2039]             #test_crop_size
