from collections import deque

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, classification_report, f1_score, accuracy_score
import tools
import seaborn as sns

def train_epoch(
        config,
        model,
        optimizer,
        device,
        loss_fn,
        train_dataloader,
        val_dataloader=None,
        epochs=10,
        scheduler=None
):

    # Tracking best validation accuracy
    best_f1 = float('-inf')
    stop_time = 0

    # Start training loop
    print("Start training...\n")
    print("-" * 60)

    optimizer.zero_grad()
    optimizer.step()
    train_val_loss = []
    for epoch_i in range(1, epochs+1):
        total_loss = []

        model = model.train()

        curr_lr = optimizer.param_groups[0]['lr']
        if scheduler and curr_lr > config.min_lr:
            scheduler.step(epoch_i)
        print(epoch_i, curr_lr)

        for batch in tqdm(train_dataloader):
            features = batch['input'].to(device)
            targets = batch['target'].to(device)
            if config.use_time_feature:
                pass
                inp_mask = batch['inp_mask'].to(device)
                # tar_mask = batch['tar_mask'].to(device)
            if config.selection_mode:
                outputs, reg = model(features[:,:,3:])
            else:
                if config.use_time_feature:
                    if model.model_name == 'dlinear':
                        outputs = model(features)
                        outputs = outputs[:, -config.forecast_horizon:]
                    else:
                        # outputs = model(features[:,:,3:], inp_mask, tar_mask)[0]
                        # outputs = outputs[:, -config.forecast_horizon:].squeeze()
                        # outputs = model(features[:,:,1:])
                        outputs, _ = model(features[:,:,3:], inp_mask)
                else:
                    if config.name_dataset == 'fi2010':
                        outputs, _ = model(features)
                    else:
                        outputs, _ = model(features[:, :, 3:])
            if config.name_dataset == 'fi2010' or not config.regression:
                outputs = torch.softmax(outputs, dim=-1)

            if config.use_time_feature:
                # targets = targets[:, -config.forecast_horizon:, -1].squeeze()
                loss = loss_fn(outputs, targets)
            else:

                loss = loss_fn(outputs, targets)

            if config.selection_mode:
                reg_lambda = epoch_i*config.reg_factor/config.anneal if epoch_i<=config.anneal else config.reg_factor
                reg_loss = reg_lambda * reg
                loss += reg_loss


            # l1_lambda = 0.00001
            # for name, param in model.named_parameters():
            #     if name == 'feature_select':
            #         fs_weight = param
            # l1_loss = l1_lambda * torch.abs(torch.sum(fs_weight) - 1)

            # loss += l1_loss

            total_loss.append(loss.item())
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
        avg_train_loss = np.mean(total_loss)
        if config.selection_mode:
            for name, param in model.named_parameters():
                if name == 'p_logit':
                    sas = torch.sigmoid(param.detach())
                    print(torch.max(sas),torch.min(sas),torch.sum(sas))
        if config.select_fun == 'FS':
            for name, param in model.named_parameters():
                if name == 'feature_select':
                    sas = torch.sigmoid(param.detach())
                    print(torch.max(sas),torch.min(sas),torch.sum(sas))

        # evaluation
        if val_dataloader is not None:

            avg_val_loss, r2 = evaluate(model, val_dataloader, device, loss_fn, config)
            # Track the best accuracy
            score = max(r2) + np.mean(np.clip(r2,-0.2, 1)) if config.name_dataset =='crypto' and config.regression else r2
            if score >= best_f1:
                best_f1 = score

                if config.name_dataset == 'crypto' and config.regression:
                    fig = plt.figure()
                    plt.plot([i for i in range(1, config.forecast_horizon + 1, 1)],
                             np.clip(r2, -50, 1), 'o-', color='g')
                    fig.suptitle(str(epoch_i) + ": " +str(max(r2)))
                    plt.show()

                # torch.save(model.state_dict(), 'best_model_state.bin')
                model.save()
                stop_time = 0
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 4:
                    print(f"Training complete! Best r2: {best_f1:.2f}%.")
                    tools.print_loss_graph(train_val_loss)
                    return
            print([epoch_i, avg_train_loss, avg_val_loss, np.round(r2, 5)])
            train_val_loss.append([avg_train_loss, avg_val_loss])
        model.save(last=True)
        print("\n")
        print(best_f1)
    tools.print_loss_graph(train_val_loss)


def evaluate(model, val_dataloader, device, loss_fn, config):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_loss = []
    pred_all = []
    lable_all = []
    source_all = []
    prob_all = []
    att_all = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Load batch to GPU
            features = batch['input'].to(device)
            targets = batch['target'].to(device)

            if config.use_time_feature:
                pass
                inp_mask = batch['inp_mask'].to(device)
                # tar_mask = batch['tar_mask'].to(device)

            if config.selection_mode:
                outputs = model(features[:,:,3:])
            else:
                if config.use_time_feature:
                    if model.model_name == 'dlinear':
                        outputs = model(features)
                        outputs = outputs[:, -config.forecast_horizon:]
                    else:
                        # outputs = model(features[:,:,3:], inp_mask, tar_mask)[0]
                        outputs, atts = model(features[:,:,3:], inp_mask)

                        # att_all.append(torch.mean(atts[0].detach(), dim=[0, 1]).cpu().numpy())
                        # outputs = model(features[:,:,1:])
                        # outputs = outputs[:, -config.forecast_horizon:].squeeze()

                else:
                    if config.name_dataset == 'fi2010':
                        outputs, _ = model(features)
                    else:
                        outputs, _ = model(features[:,:,3:])
                    # att_all.append(torch.mean(atts[0].detach(), dim=[0, 1]).cpu().numpy())

            if not config.regression:
                outputs = torch.softmax(outputs, dim=-1)

            if config.use_time_feature:
                # targets = targets[:, -config.forecast_horizon:, -1].squeeze()
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs, targets)
                
            val_loss.append(loss.item())

            # Calculate the r2
            if config.name_dataset =='crypto' and config.regression:
                pred = outputs.detach().cpu().numpy()
                target = targets.detach().cpu().numpy()
            else:
                pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                prob = torch.max(outputs, dim=1)[0].detach().cpu().numpy()
                target = targets.detach().cpu().numpy()

            pred_all.append(pred)
            if not config.regression:
                prob_all.append(prob)
            lable_all.append(target)
            source_all.append(features[:, -1, :3].detach().cpu().numpy())

        # Compute the average accuracy and loss over the validation set.
        pred_all = np.concatenate(pred_all, axis=0)
        if not config.regression:
            prob_all = np.concatenate(prob_all, axis=0)
        lable_all = np.concatenate(lable_all, axis=0)
        source_all = np.concatenate(source_all, axis=0)
        # att_all = np.mean(att_all, axis=0)
        # print(att_all.shape)
        # fig, ax = plt.subplots(1, 1, figsize=(24, 10))
        # sns.heatmap(att_all, annot=False, linewidths=0, cmap='gray', robust=True)
        # plt.show()
        # # top_fea = np.mean(att_all, axis=0).reshape(-1)
        # top_fea = att_all.reshape(21)
        # top_k = top_fea.argsort()[-50:][::-1]
        # min_k = top_fea.argsort()[:50]
        # print(top_k)
        # print(min_k)
        # print(top_fea[top_k])
        # print(top_fea[min_k])

        # print(pred_all.shape)
        # print(lable_all.shape)
        # print(source_all.shape)
        if config.plot_forecast:
            if config.regression:
                begin, end = len(pred_all)//2, len(pred_all)//2 + 1000
                forecast_values = np.array(pred_all[begin:end])
                true_values = np.array(lable_all[begin:end])
                forecast_List = []
                true_list = []
                for forecast_idx in range(0, len(true_values), config.forecast_horizon):
                    forecast_List = np.concatenate((forecast_List, forecast_values[forecast_idx, :]))
                    true_list = np.concatenate((true_list, true_values[forecast_idx, :]))
                fig, ax = plt.subplots(1, 1, figsize=(18, 8))
                # ax.plot(true_values[:, 0], label='truth', alpha=0.3, color='black')
                # ax.plot(forecast_values[:, 0], label='forecast', alpha=0.8)
                ax.plot(true_list.reshape(-1), label='truth', alpha=0.3, color='black')
                ax.plot(forecast_List.reshape(-1), label='forecast', alpha=0.8, linestyle='dashed')
                ax.set_xlim(left=0, right=len(true_list))
                Min, Max = min(true_list), max(true_list)
                index = np.arange(0, end - begin, config.forecast_horizon)
                ax.vlines(x=index, colors='red',
                          ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                plt.show()

            else:
                begin, end = len(pred_all)//2, len(pred_all)//2 + 5000
                forecast_values = np.array(pred_all[begin:end])
                forecast_prob = np.array(prob_all[begin:end])
                positive_prob = np.array(prob_all[begin:end])
                negative_prob = np.array(prob_all[begin:end])
                positive_prob = positive_prob[forecast_values == 2]
                negative_prob = negative_prob[forecast_values == 1]

                buy_throd = np.quantile(positive_prob, 0.2)
                sell_throd = np.quantile(negative_prob, 0.2)

                true_values = np.array(lable_all[begin:end])
                prices = np.array(source_all)[begin:end, 0]
                prices = (prices/prices[0])
                Min, Max = min(prices), max(prices)
                index = np.arange(0, end-begin)

                fig, ax = plt.subplots(1, 1, figsize=(24, 10))
                ax.plot(prices, label='price', alpha=1, color='black')
                # ax.plot(forecast_prob, label='prob', alpha=0.2, color='orange')
                ax.vlines(x=index[np.logical_and(forecast_values == 2,forecast_prob >=buy_throd)], colors='red',
                          ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                ax.vlines(x=index[np.logical_and(forecast_values == 1,forecast_prob >=sell_throd)], colors='green', ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)

                plt.show()

                fig, ax = plt.subplots(1, 1, figsize=(24, 10))
                ax.plot(prices, label='price', alpha=1, color='black')
                ax.vlines(x=index[true_values==2], colors='red',
                          ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                ax.vlines(x=index[true_values==1], colors='green', ymin=Min,
                          ymax=Max
                          , linewidths=0.1, alpha=0.6)
                plt.show()

        if config.backtesting:
            forecast_values = np.array(pred_all)
            forecast_prob = np.array(prob_all)
            positive_prob = np.array(prob_all)
            negative_prob = np.array(prob_all)
            positive_prob = positive_prob[forecast_values == 2]
            negative_prob = negative_prob[forecast_values == 1]
            buy_throd = np.quantile(positive_prob, 0.7)
            sell_throd = np.quantile(negative_prob, 0.7)
            short = True

            mid_price = np.array(source_all)[:, 0]
            bid = np.array(source_all)[:, 1]
            ask = np.array(source_all)[:, 2]
            print(bid[0], ask[0])
            trade_delay = 1
            trade_fee = 0.02 # 0.02%
            # trade_fee = 0.0 # 0.02%
            DL_machine = tools.tradebot(volume=1, fee=trade_fee, short=short)
            Fallow_machine = tools.tradebot(volume=1, fee=trade_fee, short=short)
            basicline_signal = 0.0003
            curr_price = mid_price[0]
            for i in range(len(forecast_values)-trade_delay):



                if forecast_values[i] == 2 and forecast_prob[i] >= buy_throd:
                    DL_machine.buy_signal(ask[i+trade_delay], bid[i+trade_delay])
                elif forecast_values[i] == 1 and forecast_prob[i] >= sell_throd:
                    DL_machine.sell_signal(ask[i+trade_delay], bid[i+trade_delay])
                else:
                    DL_machine.add_his()

                if (mid_price[i]/curr_price - 1) > basicline_signal:
                    Fallow_machine.buy_signal(ask[i+trade_delay], bid[i+trade_delay])
                elif -(mid_price[i]/curr_price - 1) > basicline_signal:
                    Fallow_machine.sell_signal(ask[i+trade_delay], bid[i+trade_delay])
                else:
                    Fallow_machine.add_his()
                curr_price = mid_price[i]

            print([np.quantile(positive_prob, i / 10) for i in range(1, 10, 1)])
            print([np.quantile(negative_prob, i / 10) for i in range(1, 10, 1)])
            print('buy_throd', buy_throd)
            print('sell_throd', sell_throd)

            DL_his = DL_machine.get_his()
            FL_his = Fallow_machine.get_his()
            print('trade time:', DL_machine.get_trade_time())
            print('winning rate:', DL_machine.get_winning_rate())
            print('meaningful winning:', DL_machine.get_underfee_rate())
            print('max drawback:', DL_machine.get_max_drawdown())
            print('highest value:', max(DL_his))
            print('profit:', DL_his[-1])
            print('buy_and_hold:', (mid_price/mid_price[0] - 1)[len(DL_his)-1])
            print('baseLine profit', FL_his[-1])
            print('baseLine max drawback', Fallow_machine.get_max_drawdown())
            fig, ax = plt.subplots(1, 1, figsize=(24, 10))
            ax.plot(mid_price/mid_price[0] -1, label='price', alpha=1, color='black')
            ax.plot((np.array(DL_his)), label='profit', alpha=0.7, color='red')
            ax.plot(forecast_prob/10, label='forecast_prob', alpha=0.7, color='green')
            ax.set_xlim(left=0, right=len(DL_his))
            plt.show()
            # b, w, d = pred_all.shape
            # pred_all = pred_all.reshape(b, w)
            # lable_all = lable_all.reshape(b, w)


    if config.name_dataset=='fi2010' or not config.regression:
        print(classification_report(lable_all, pred_all, zero_division=0, digits=5))
        print(accuracy_score(lable_all,pred_all))
        return np.mean(val_loss), f1_score(lable_all,pred_all, average='macro')
        # return np.mean(val_loss), np.mean(val_loss)

    else:
        r2 = r2_score(lable_all, pred_all, multioutput='raw_values')
        print(r2)
        # r2 = r2_previous_steps(lable_all, pred_all, source_all)
        print('Task1', np.argmax(r2), max(r2), np.mean(np.clip(r2,-50, 1)))
        return np.mean(val_loss), r2


# def pridict(model, test_dataloader, device):
#     # Put the model into the evaluation mode. The dropout layers are disabled
#     # during the test time.
#     model.eval()
#
#     # Tracking variables
#     logits = []
#     pred_all = []
#     # For each batch in our validation set...
#     with torch.no_grad():
#         for batch in tqdm(test_dataloader):
#             # Load batch to GPU
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#
#             # Compute logits
#
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#
#             # Calculate the accuracy rate
#             logits.extend(F.softmax(outputs,dim=1).cpu().numpy())
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             pred_all.extend(preds)
#     return pred_all, logits

def r2_previous_steps(true, pred, source):
    source = np.array(source)
    numerator = ((true - pred) ** 2).sum(axis=0, dtype=np.float64)
    # prev_label = np.concatenate([np.zeros((len(pred),1)), (source[:, 1:-1])], axis=1)
    prev_label = np.concatenate([np.zeros((len(pred), 1)), (true[:, 0:-1])], axis=1)
    denominator = (
            (true - (prev_label)) ** 2
    ).sum(axis=0, dtype=np.float64)

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return output_scores

