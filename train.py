try:
    import warnings
    warnings.filterwarnings("ignore")

    from os.path import join, isfile, isdir
    from queue import Empty, Queue
    from threading import Thread

    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.tensorboard import SummaryWriter
    import torch
    from tqdm import tqdm

    from libs.torchsummaryX import summary
    from contextlib import redirect_stdout

    from libs.InputGenerator import InputGenerator
    from libs.ModelsCheckpointer import *
    from libs.SGDR import CosineAnnealingWarmRestarts
    from libs.copy_tree import copytree_multi
    from libs.metrics import *
    from libs.models import *
    from libs.parse_args import *
    from libs.service_defs import *
    from types import SimpleNamespace
    from libs import kornia
    from libs.kornia.constants import Resample
    from libs.kornia.color import *
    from libs.kornia import augmentation as K
    import libs.kornia.augmentation.functional as KF
    import libs.kornia.augmentation.random_generator as rg
except ImportError as e:
    print(e)
    raise ImportError



class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
  """Threaded worker for pre-processing input data.
  tokill is a thread_killer object that indicates whether a thread should be terminated
  dataset_generator is the training/validation dataset generator
  batches_queue is a limited size thread-safe Queue instance.
  """
  while tokill() == False:
    for batch, ((batch_images, batch_masks), batch_targets) in enumerate(dataset_generator):
        #We fill the queue with new fetched batch until we reach the max       size.
        batches_queue.put((batch, ((batch_images, batch_masks), batch_targets)), block=True)
        if tokill() == True:
          return


def threaded_cuda_batches(tokill,cuda_batches_queue,batches_queue):
  """Thread worker for transferring pytorch tensors into
  GPU. batches_queue is the queue that fetches numpy cpu tensors.
  cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
  """
  while tokill() == False:
    batch, ((batch_images, batch_masks), batch_targets) = batches_queue.get(block=True)
    batch_images_np = np.transpose(batch_images, (0, 3, 1, 2))
    batch_masks_np = np.transpose(batch_masks, (0, 3, 1, 2))

    batch_images = torch.from_numpy(batch_images_np)
    batch_masks = torch.from_numpy(batch_masks_np)
    batch_labels = torch.from_numpy(batch_targets)

    batch_images = Variable(batch_images).cuda()
    batch_masks = Variable(batch_masks).cuda()
    batch_labels = Variable(batch_labels).cuda()

    cuda_batches_queue.put((batch, ((batch_images, batch_masks), batch_labels)), block=True)
    if tokill() == True:
      return





def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cuda_batches_queue: Queue,
                loss_fn: callable, metrics: list, TBwriter: SummaryWriter, STEPS_PER_EPOCH: int,
                ElasticTransformer_instance: ElasticTransformer, args):
    model.train()
    term_columns = os.get_terminal_size().columns
    pbar = tqdm(total=STEPS_PER_EPOCH, ncols=min(term_columns,180))
    metrics_values = dict([(m.name, 0.0) for m in metrics])

    # activations = []

    for batch_idx in range(STEPS_PER_EPOCH):
        _, ((img, msk), trg) = cuda_batches_queue.get(block=True)

        flow = ElasticTransformer_instance.generate_flow()
        affine_params = rg.random_affine_generator(args.batch_size, img.shape[-2], img.shape[-1],
                                                   degrees=torch.from_numpy(np.array([-45.0, 45.0])),
                                                   scale=torch.from_numpy(np.array([0.7, 1.0])),
                                                   same_on_batch=False)
        for k in affine_params.keys():
            affine_params[k] = affine_params[k].cuda()

        img = ElasticTransformer_instance.transform(img, flow)
        img = KF.apply_affine(img, affine_params)
        img = img/255.0
        img = torch.clamp(img, 0.0, 1.0)

        msk = ElasticTransformer_instance.transform(msk, flow)
        msk = KF.apply_affine(msk, affine_params)
        msk = msk/255.0
        msk = (msk >= 0.5).float()


        optimizer.zero_grad()
        if ((batch_idx == STEPS_PER_EPOCH-1) & args.debug):
            output, activations = model(img, msk, dump_activations=True)
        else:
            output = model(img, msk)
        loss = loss_fn(output, trg)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        for m in metrics:
            metrics_values[m.name] = metrics_values[m.name] + m(output, trg)

        pbar.update(1)
        pbar.set_postfix_str('Train Epoch: %d [%d/%d (%.2f%%)]\tLoss: %.6f; accuracy: %.6f; leq1_accuracy: %.6f' % (epoch, batch_idx+1, STEPS_PER_EPOCH,
                                                                                                                    100. *(batch_idx+1) / STEPS_PER_EPOCH,
                                                                                                                    loss.item(),
                                                                                                                    metrics_values['accuracy']/(batch_idx+1),
                                                                                                                    metrics_values['leq1_accuracy']/(batch_idx+1)))
        # print()
        if batch_idx >= STEPS_PER_EPOCH-1:
            break

    for tag, param in model.named_parameters():
        TBwriter.add_histogram('grad/%s'%tag, param.grad.data.cpu().numpy(), epoch)
        TBwriter.add_histogram('weight/%s' % tag, param.data.cpu().numpy(), epoch)

    # if (args.debug & (epoch % 10 == 0)):
    #     TBwriter.add_images('input', img, epoch)

    if args.debug:
        for mname in activations.keys():
            TBwriter.add_histogram('activations/%s' % mname, activations[mname], epoch)

    for m in metrics:
        metrics_values[m.name] = float(((metrics_values[m.name] / STEPS_PER_EPOCH)).cpu())

    pbar.set_postfix_str('Train Epoch: %d; Loss: %.6f; accuracy: %.6f; leq1_accuracy: %.6f' % (epoch,
                                                                                               loss.item(),
                                                                                               metrics_values['accuracy'],
                                                                                               metrics_values['leq1_accuracy']))

    pbar.close()

    losses_dict = {'train_loss': loss.item()}
    return dict(**losses_dict, **metrics_values)



def validation(model: nn.Module, cuda_batches_queue: Queue, loss_fn: callable, metrics: list,
               TBwriter: SummaryWriter, VAL_STEPS: int, epoch: int,
               ElasticTransformer_instance: ElasticTransformer, args):
    model.eval()
    val_loss = 0
    metrics_values = dict([(m.name, 0.0) for m in metrics])
    with torch.no_grad():
        term_columns = os.get_terminal_size().columns
        pbar = tqdm(total=VAL_STEPS, ncols=min(term_columns, 180))
        for batch_idx in range(VAL_STEPS):
            _, ((img, msk), trg) = cuda_batches_queue.get(block=True)


            if args.tta:
                flow = ElasticTransformer_instance.generate_flow()
                affine_params = rg.random_affine_generator(args.batch_size, img.shape[-2], img.shape[-1],
                                                           degrees=torch.from_numpy(np.array([-45.0, 45.0])),
                                                           scale=torch.from_numpy(np.array([0.7, 1.0])),
                                                           same_on_batch=False)
                for k in affine_params.keys():
                    affine_params[k] = affine_params[k].cuda()

                img = ElasticTransformer_instance.transform(img, flow)
                img = img / 255.0
                img = KF.apply_affine(img, affine_params)
                img = torch.clamp(img, 0.0, 1.0)

                msk = ElasticTransformer_instance.transform(msk, flow)
                msk = KF.apply_affine(msk, affine_params)
                msk = msk / 255.0
                msk = (msk >= 0.5).float()
            else:
                img = img / 255.0
                img = torch.clamp(img, 0.0, 1.0)
                msk = msk / 255.0
                msk = (msk >= 0.5).float()


            if ((batch_idx == VAL_STEPS-1) & args.debug):
                output, activations = model(img, msk, dump_activations=True)
            else:
                output = model(img, msk)
            val_loss += loss_fn(output, trg).item()

            for m in metrics:
                metrics_values[m.name] = metrics_values[m.name] + m(output, trg)

            pbar.update(1)
            if batch_idx >= VAL_STEPS-1:
                break
        pbar.close()


    # if (args.debug & (epoch % 10 == 0)):
    #     TBwriter.add_images('input', img, epoch)

    if args.debug:
        for mname in activations.keys():
            TBwriter.add_histogram('activations/%s' % mname, activations[mname], epoch)

    for m in metrics:
        metrics_values[m.name] = float((metrics_values[m.name] / VAL_STEPS).cpu())

    val_loss = val_loss/VAL_STEPS
    losses_dict = {'val_loss': val_loss}
    return dict(**losses_dict, **metrics_values)







def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    torch.autograd.set_detect_anomaly(True)

    resume_state = None
    if 'resume' in args:
        # restore epoch and other parameters
        with open(os.path.join('./', 'scripts_backup', args.resume, 'launch_parameters.txt'), 'r') as f:
            args_resume = f.readlines()[1:]
            args_resume = [t.replace('\n', '') for t in args_resume]
            args_resume = parse_args(args_resume)
            for k in [k for k in args_resume.__dict__.keys()]:
                if k in ['run_name', 'snapshot', 'resume', 'lr']:
                    continue
                # if k in args.__dict__.keys():
                #     continue
                args.__dict__[k] = getattr(args_resume, k)

        resume_state = SimpleNamespace()
        resume_state.dates_train = np.load(os.path.join('./', 'scripts_backup', args.resume, 'dates_train.npy'), allow_pickle=True)
        resume_state.dates_val = np.load(os.path.join('./', 'scripts_backup', args.resume, 'dates_val.npy'), allow_pickle=True)
        resume_state.epoch_snapshot = find_files(os.path.join('./logs', args.resume), 'ep????.pth.tar')[0]
        resume_state.epoch = int(os.path.basename(resume_state.epoch_snapshot).replace('.pth.tar', '').replace('ep', ''))
        resume_state.lr = args.lr


    #region args parsing
    curr_run_name = args.run_name
    EPOCHS = args.epochs
    if 'steps_per_epoch' in args:
        STEPS_PER_EPOCH = args.steps_per_epoch
    else:
        STEPS_PER_EPOCH = None

    if 'val_steps' in args:
        VAL_STEPS = args.val_steps
    else:
        VAL_STEPS = None
    #endregion

    #region preparations
    base_logs_dir = os.path.join('./logs', curr_run_name)
    try:
        EnsureDirectoryExists(base_logs_dir)
    except:
        print(f'logs directory couldn`t be found and couldn`t be created:\n{base_logs_dir}')
        raise FileNotFoundError(f'logs directory couldn`t be found and couldn`t be created:\n{base_logs_dir}')

    scripts_backup_dir = os.path.join('./scripts_backup', curr_run_name)
    try:
        EnsureDirectoryExists(scripts_backup_dir)
    except:
        print(f'backup directory couldn`t be found and couldn`t be created:\n{scripts_backup_dir}')
        raise FileNotFoundError(f'backup directory couldn`t be found and couldn`t be created:\n{scripts_backup_dir}')

    tboard_dir_train = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'TBoard', 'train')
    tboard_dir_val = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'TBoard', 'val')
    try:
        EnsureDirectoryExists(tboard_dir_train)
    except:
        print('Tensorboard directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir_train)
        raise FileNotFoundError(
            'Tensorboard directory directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir_train)
    try:
        EnsureDirectoryExists(tboard_dir_val)
    except:
        print('Tensorboard directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir_val)
        raise FileNotFoundError(
            'Tensorboard directory directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir_val)
    #endregion

    # region backing up the scripts configuration
    print('backing up the scripts')
    ignore_func = lambda dir, files: [f for f in files if (isfile(join(dir, f)) and f[-3:] != '.py')] + [d for d in files if ((isdir(d)) & (('srcdata' in d) |
                                                                                                                                            ('scripts_backup' in d) |
                                                                                                                                            ('__pycache__' in d) |
                                                                                                                                            ('.pytest_cache' in d) |
                                                                                                                                            d.endswith('.ipynb_checkpoints') |
                                                                                                                                            d.endswith('logs.bak') |
                                                                                                                                            d.endswith('outputs') |
                                                                                                                                            d.endswith('processed_data') |
                                                                                                                                            d.endswith('build') |
                                                                                                                                            d.endswith('logs') |
                                                                                                                                            d.endswith('snapshots')))]
    copytree_multi('./',
                   './scripts_backup/%s/' % curr_run_name,
                   ignore=ignore_func)

    with open(os.path.join(scripts_backup_dir, 'launch_parameters.txt'), 'w+') as f:
        f.writelines([f'{s}\n' for s in sys.argv])
    # endregion backing up the scripts configuration



    cuda = True if torch.cuda.is_available() else False
    if cuda:
        torch.cuda.set_device(0)
    cuda_dev = torch.device('cuda:0')



    # if 'srcdata_list' in args.__dict__.keys():
    #     data_index_fname = args.srcdata_list
    # else:
    #     data_index_fname = './srcdata/dts_img-fnames_msk-fnames_TCC8_dtLEQ300sec.pkl'
    train_index_fname = args.train_list
    test_index_fname = args.test_list


    print('creating the model')
    if args.pnet:
        model = SIAmodel_PyramidNet(args, classes_num=9)
    else:
        model = SIAmodel(args, classes_num=9)
    if resume_state is not None:
        model.load_state_dict(torch.load(resume_state.epoch_snapshot))

    TB_writer_train = SummaryWriter(log_dir=tboard_dir_train)
    TB_writer_val = SummaryWriter(log_dir=tboard_dir_val)

    model = model.cuda()

    print('logging the graph of the model')
    TB_writer_train.add_graph(model, [torch.tensor(np.random.random(size=(args.batch_size, 3, args.img_size, args.img_size)).astype(np.float32)).cuda(),
                                      torch.tensor(np.random.random(size=(args.batch_size, 3, args.img_size, args.img_size)).astype(np.float32)).cuda()])

    print('logging the summary of the model')
    with open(os.path.join(base_logs_dir, 'model_structure.txt'), 'w') as f:
        with redirect_stdout(f):
            summary(model,
                    x = torch.tensor(np.random.random(size=(args.batch_size, 3, args.img_size, args.img_size)).astype(np.float32)).cuda(),
                    msk = torch.tensor(np.random.random(size=(args.batch_size, 3, args.img_size, args.img_size)).astype(np.float32)).cuda())



    if args.model_only:
        quit()

    if resume_state is not None:
        optimizer = optim.Adam(model.parameters(), lr=resume_state.lr, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS-resume_state.epoch, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)
    else:
        optimizer = optim.Adam(model.parameters(), lr=3.0e-4, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=128, T_mult=2, eta_min=1.0e-9, lr_decay=0.75)


    #region train dataset
    # if resume_state is not None:
    #     subsetting_option = resume_state.dates_val
    # else:
    #     subsetting_option = 0.75
    train_ds = InputGenerator(data_index_fname=args.train_list,
                              args = args,
                              batch_size=args.batch_size,
                              img_size = (3,args.img_size,args.img_size),
                              # subsetting_option=subsetting_option,
                              model_type=args.model_type,
                              rebalance = True,
                              debug = False)
    # dates_used_train = train_ds.dates_used
    # np.save(os.path.join(scripts_backup_dir, 'dates_train.npy'), dates_used_train)
    batches_queue_length = 16 if args.memcache else STEPS_PER_EPOCH
    batches_queue_length = min(batches_queue_length, 64)
    train_batches_queue = Queue(maxsize=batches_queue_length)
    train_cuda_batches_queue = Queue(maxsize=4)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)
    preprocess_workers = 4
    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(train_thread_killer, train_batches_queue, train_ds))
        thr.start()
    train_cuda_transfers_thread_killer = thread_killer()
    train_cuda_transfers_thread_killer.set_tokill(False)
    train_cudathread = Thread(target=threaded_cuda_batches, args=(train_cuda_transfers_thread_killer, train_cuda_batches_queue, train_batches_queue))
    train_cudathread.start()
    #endregion train dataset

    # region test dataset
    val_ds = InputGenerator(data_index_fname=args.test_list,
                            args=args,
                            batch_size=args.val_batch_size,
                            img_size=(3, args.img_size, args.img_size),
                            # subsetting_option=dates_used_train,
                            model_type=args.model_type,
                            rebalance=True,
                            debug=False)
    # dates_used_val = val_ds.dates_used
    # np.save(os.path.join(scripts_backup_dir, 'dates_val.npy'), dates_used_val)
    batches_queue_length = 16 if args.memcache else VAL_STEPS
    batches_queue_length = min(batches_queue_length, 64)
    val_batches_queue = Queue(maxsize=batches_queue_length)
    val_cuda_batches_queue = Queue(maxsize=4)
    val_thread_killer = thread_killer()
    val_thread_killer.set_tokill(False)
    for _ in range(preprocess_workers):
        thr = Thread(target=threaded_batches_feeder, args=(val_thread_killer, val_batches_queue, val_ds))
        thr.start()
    val_cuda_transfers_thread_killer = thread_killer()
    val_cuda_transfers_thread_killer.set_tokill(False)
    val_cudathread = Thread(target=threaded_cuda_batches, args=(val_cuda_transfers_thread_killer, val_cuda_batches_queue, val_batches_queue))
    val_cudathread.start()
    # endregion train dataset



    ET = ElasticTransformer(img_size=(3,args.img_size,args.img_size),
                            batch_size=args.batch_size,
                            flow_initial_size=(args.img_size//32, args.img_size//32),
                            flow_displacement_range=args.img_size/32)

    if args.model_type == 'PC':
        def cross_entropy(pred, soft_targets):
            log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
            return torch.mean(torch.sum(- soft_targets * log_softmax_pred, 1))

        loss_fn = cross_entropy
    elif args.model_type == 'OR':
        loss_fn = F.binary_cross_entropy

    metric_equal = accuracy(name='accuracy', model_type=args.model_type, batch_size=args.batch_size)
    metric_leq1 = diff_leq_accuracy(name='leq1_accuracy', model_type=args.model_type, batch_size=args.batch_size, leq_threshold=1)


    #region creating checkpoint writers
    val_loss_checkpointer = ModelsCheckpointer(model, 'ep%04d_valloss_%.6e.pth.tar', ['epoch', 'val_loss'],
                                               base_dir = base_logs_dir, replace=True,
                                               watch_metric_names=['val_loss'], watch_conditions=['min'])
    val_accuracy_checkpointer = ModelsCheckpointer(model, 'ep%04d_valacc_%.6e.pth.tar', ['epoch', 'accuracy'],
                                                   base_dir=base_logs_dir, replace=True,
                                                   watch_metric_names=['accuracy'], watch_conditions=['max'])
    val_leq1_accuracy_checkpointer = ModelsCheckpointer(model, 'ep%04d_valleq1acc_%.6e.pth.tar', ['epoch', 'leq1_accuracy'],
                                                        base_dir=base_logs_dir, replace=True,
                                                        watch_metric_names=['leq1_accuracy'], watch_conditions=['max'])
    mandatory_checkpointer = ModelsCheckpointer(model, 'ep%04d.pth.tar', ['epoch'], base_dir=base_logs_dir, replace=True)

    checkpoint_saver_final = ModelsCheckpointer(model, 'final.pth.tar', [], base_dir=base_logs_dir, replace=False)
    #endregion



    print('\n\nstart training')
    start_epoch = 1 if resume_state is None else resume_state.epoch
    for epoch in range(start_epoch, EPOCHS+1):
        print('\n\n%s: Train epoch: %d of %d' % (args.run_name, epoch, EPOCHS))
        train_metrics = train_epoch(model, optimizer, epoch, train_cuda_batches_queue, loss_fn=loss_fn,
                                    metrics=[metric_equal, metric_leq1], TBwriter=TB_writer_train,
                                    STEPS_PER_EPOCH=STEPS_PER_EPOCH,
                                    ElasticTransformer_instance=ET, args=args)
        print(str(train_metrics))
        print('\nValidation:')
        val_metrics = validation(model, val_cuda_batches_queue, loss_fn=loss_fn, metrics=[metric_equal, metric_leq1],
                                 TBwriter=TB_writer_val, VAL_STEPS=VAL_STEPS, epoch=epoch,
                                 ElasticTransformer_instance=ET, args=args)
        print(str(val_metrics))

        # note: this re-shuffling will not make an immediate effect since the queues are already filled with the
        # examples from the previous shuffle-states of datasets
        train_ds.shuffle()
        val_ds.shuffle()

        #region checkpoints
        val_loss_checkpointer.save_models(pdict={'epoch': epoch, 'val_loss': val_metrics['val_loss']},
                                          metrics=val_metrics)
        val_accuracy_checkpointer.save_models(pdict={'epoch': epoch, 'accuracy': val_metrics['accuracy']},
                                              metrics=val_metrics)
        val_leq1_accuracy_checkpointer.save_models(pdict={'epoch': epoch, 'leq1_accuracy': val_metrics['leq1_accuracy']},
                                                   metrics=val_metrics)
        mandatory_checkpointer.save_models(pdict={'epoch': epoch})
        #endregion

        # region write losses to tensorboard
        TB_writer_train.add_scalar('loss', train_metrics['train_loss'], epoch)
        TB_writer_train.add_scalar('LR', scheduler.get_last_lr()[-1], epoch)
        TB_writer_train.add_scalar('accuracy', train_metrics['accuracy'], epoch)
        TB_writer_train.add_scalar('leq1_accuracy', train_metrics['leq1_accuracy'], epoch)

        TB_writer_val.add_scalar('accuracy', val_metrics['accuracy'], epoch)
        TB_writer_val.add_scalar('loss', val_metrics['val_loss'], epoch)
        TB_writer_val.add_scalar('leq1_accuracy', val_metrics['leq1_accuracy'], epoch)
        # endregion
        scheduler.step(epoch=epoch)

    checkpoint_saver_final.save_models(None)


    # train_ds.close()
    # test_ds.close()
    train_thread_killer.set_tokill(True)
    train_cuda_transfers_thread_killer.set_tokill(True)
    val_thread_killer.set_tokill(True)
    val_cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            train_cuda_batches_queue.get(block=True, timeout=1)
            val_batches_queue.get(block=True, timeout=1)
            val_cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass


if __name__ == "__main__":
    main()
