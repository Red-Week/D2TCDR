import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for epoch_temp in range(epochs):

        model_joint.train()
        flag_update = 0
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            target=list(batch['next'].values())
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            if pretrain_flag:
                con_batch = con_data.sample(n=args.batch_size).to_dict()
                con_seq = list(con_batch['seq'].values())
                con_seq = torch.LongTensor(con_seq)
                con_seq = con_seq.to(device)
            else:
                con_seq = None
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, em_loss = model_joint(seq, target, con_seq, pretrain_flag, args, epoch_temp, train_flag=True)  
            loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, target, pretrain_flag)  ## use this not above        
            loss_all = loss_diffu_value + em_loss*args.loss_lambda
            loss_all.backward()
        
            optimizer.step()
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        lr_scheduler.step()
        
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                for j in range(int(val_data.shape[0]/args.batch_size)):
                    batch = val_data[j * args.batch_size: (j + 1)* args.batch_size].to_dict()
                    seq = list(batch['seq'].values())
                    target=list(batch['next'].values())
                    seq = torch.LongTensor(seq)
                    target = (torch.LongTensor(target)).unsqueeze(1)
                    seq = seq.to(device)
                    target = target.to(device)
                    scores_rec, rep_diffu, _, _, _, _ , _ = model_joint(seq, target, None, pretrain_flag, args, epoch_temp, train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu, pretrain_flag)    ### inner_production
                    output = torch.full_like(scores_rec_diffu, -100.0)
                    row_indices = torch.arange(args.batch_size).unsqueeze(1)
                    negative_samples = list(batch['negative_samples'].values())
                    output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
                    scores_rec_diffu = output
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
                        
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update += 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                if flag_update >= 3:
                    best_model = copy.deepcopy(model_joint)
                    if pretrain_flag:
                        torch.save(model_joint.state_dict(), "./saved_model/"+args.s_dataset + "_" + args.t_dataset+"/model.pth")
            if bad_count >= args.patience:
                break
          
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for j in range(int(test_data.shape[0]/args.batch_size)):
            batch = test_data[j * args.batch_size: (j + 1)* args.batch_size].to_dict()
            seq = list(batch['seq'].values())
            target=list(batch['next'].values())
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            scores_rec, rep_diffu, _, _, _, _, _ = best_model(seq, target, None, pretrain_flag, args, 0.2, train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu,pretrain_flag)   ### Inner Production
            output = torch.full_like(scores_rec_diffu, -100.0)
            row_indices = torch.arange(args.batch_size).unsqueeze(1)
            negative_samples = list(batch['negative_samples'].values())
            output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
            scores_rec_diffu = output
            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    return best_model, test_metrics_dict_mean
    
