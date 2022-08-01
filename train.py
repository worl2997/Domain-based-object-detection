import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.parse_config import parse_data_cfg

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *



hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/416 if img_size != 416)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train(args,model_cfg, device, tb_writer, path, mixed_precision):
    wdir =  os.path.join(path.model_save_path, args.domain)
    backup_wdir = os.path.join(wdir, 'backup')
    createFolder(backup_wdir)
    createFolder(wdir)
    # last = wdir + 'last.pt'
    # best = wdir + 'best.pt'
    # results_file = 'results.txt'

    cfg = model_cfg
    img_size = args.img_size

    last = os.path.join(wdir, args.domain + str(args.classes) + '_'+ args.model + '_last.pt')
    best = os.path.join(wdir, args.domain + str(args.classes)+ '_'+ args.model + '_best.pt')
    results_file = os.path.join(wdir, args.domain + str(args.classes)+'_'+ args.model + '_results.txt')

    args.weights = last if args.resume else args.weights

    epochs = 1 if args.prebias else args.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = args.batch_size
    accumulate = args.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = last if args.resume else args.weights  # initial training weights
    data = os.path.join(path.DATA_FILE_DIR, args.domain + '.data')  # args.data
    data_dict = parse_data_cfg(data)


    if 'pw' not in args.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    # Initialize
    init_seeds()
    if args.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=args.arc).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    if args.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = float('inf')
    if weights:
        attempt_download(weights)

        if weights.endswith('.pt'):  # pytorch format
            # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(weights, map_location=device)

            # load model
            try:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.cfg, args.weights)
                raise KeyError(s) from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            cutoff = load_darknet_weights(model, weights)

        if args.transfer or args.prebias:  # transfer learning edge (yolo) layers
            nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

            if args.prebias:
                for p in optimizer.param_groups:
                    # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                    p['lr'] *= 100  # lr gain
                    if p.get('momentum') is not None:  # for SGD but not Adam
                        p['momentum'] *= 0.9

            for p in model.parameters():
                if args.prebias and p.numel() == nf:  # train (yolo biases)
                    p.requires_grad = True
                elif args.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                    p.requires_grad = True
                else:  # freeze layer
                    p.requires_grad = False

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        except:
            print('mixed precision error')
    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=args.rect,  # rectangular training
                                  image_weights=args.img_weights,
                                  cache_labels=epochs > 10,
                                  cache_images=args.cache_images and not args.prebias)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not args.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Test Dataloader
    if not args.prebias:
        testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, args.img_size, batch_size,
                                                                     hyp=hyp,
                                                                     rect=True,
                                                                     cache_labels=True,
                                                                     cache_images=args.cache_images),
                                                 batch_size=batch_size,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

    # Start training
    nb = len(dataloader)
    model.nc = nc  # attach number of classes to model
    model.arc = args.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting %s for %g epochs...' % ('prebias' if args.prebias else 'training', epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            targets = targets.to(device)

            # Multi-Scale training
            if args.multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Hyperparameter burn-in
            # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
            # if ni <= n_burn:
            #     for m in model.named_modules():
            #         if m[0].endswith('BatchNorm2d'):
            #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
            #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
            #     for x in optimizer.param_groups:
            #         x['lr'] = hyp['lr0'] * g
            #         x['weight_decay'] = hyp['weight_decay'] * g

            # Run model
            pred = model(imgs) # imgs shape -> [8,3,416,416]

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if args.prebias:
            print_model_biases(model)
        elif not args.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=args.img_size,
                                      model=model,
                                      conf_thres=0.001 if final_epoch else 0.1,  # 0.1 for speed
                                      save_json=final_epoch and is_coco,
                                      dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(args.name) and args.bucket and not args.prebias:
            os.system('gsutil cp results.txt gs://%s/results%s.txt' % (args.bucket, args.name))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not args.nosave) or (final_epoch and not args.evolve) or args.prebias
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 5 == 0:
                torch.save(chkpt, os.path.join(backup_wdir, args.domain + str(args.classes)+ '_'+ args.model + '_backup%g.pt' % epoch))

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    if len(args.name) and not args.prebias:
        fresults, flast, fbest = 'results%s.txt' % args.name, 'last%s.pt' % args.name, 'best%s.pt' % args.name
        os.rename('results.txt', fresults)
        os.rename(wdir + 'last.pt', wdir + flast) if os.path.exists(wdir + 'last.pt') else None
        os.rename(wdir + 'best.pt', wdir + fbest) if os.path.exists(wdir + 'best.pt') else None

        # save to cloud
        if args.bucket:
            os.system('gsutil cp %s %s gs://%s' % (fresults, wdir + flast, args.bucket))

    plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


def prebias(args,model_cfg, device, tb_writer, path, mixed_precision):
    # trains output bias layers for 1 epoch and creates new backbone
    if args.prebias:
        wdir = os.path.join(path.model_save_path, args.domain)
        last = wdir + 'last.pt'

        a = args.img_weights  # save settings
        args.img_weights = False  # disable settings

        train(args,model_cfg, device, tb_writer, path,mixed_precision)  # transfer-learn yolo biases for 1 epoch
        create_backbone(last)  # saved results as backbone.pt

        args.weights = wdir + 'backbone.pt'  # assign backbone
        args.prebias = False  # disable prebias
        args.img_weights = a  # reset settings


def train_model(args, model_cfg, path):

    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        print('mixed precision is not adapted..')
        mixed_precision = False  # not installed

    tb_writer = None
    device = torch_utils.select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    if device.type == 'cpu':
        mixed_precision = False


    # scale hyp['obj'] by img_size (evolved at 416)
    hyp['obj'] *= args.img_size / 416.

    if not args.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter()
        except:
            pass
        prebias(args,model_cfg, device, tb_writer, path,mixed_precision)  # optional
        train(args,model_cfg, device, tb_writer, path, mixed_precision)  # train normally

    else:  # Evolve hyperparameters (optional)
        args.notest = True  # only test final epoch
        args.nosave = True  # only save final checkpoint
        if args.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % args.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                x = np.loadtxt('evolve.txt', ndmin=2)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                if parent == 'single' or len(x) == 1:
                    x = x[fitness(x).argmax()]
                elif parent == 'weighted':  # weighted combination
                    n = min(10, x.shape[0])  # number to merge
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

                # Mutate
                np.random.seed(int(time.time()))
                s = np.random.random() * 0.3  # sigma
                g = [1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # gains
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn() * s * g[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            prebias(args, model_cfg, device, tb_writer, path)  # optional
            results = train(args, model_cfg, device, tb_writer, path,mixed_precision)

            # Write mutation results
            print_mutation(hyp, results, args.bucket)

            # Plot results
            # plot_evolution_results(hyp)