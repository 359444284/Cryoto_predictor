from collections import deque

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, classification_report, f1_score, accuracy_score
from utils.simple_backtest import *
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
        scheduler=None,
        save_path=None
):

    # Tracking best validation accuracy
    best_score = float('-inf') if config.regression else float('-inf')
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
                    # outputs = model(features[:,:,3:], inp_mask, tar_mask)[0]
                    outputs, _ = model(features[:,:,3:], inp_mask)
                else:
                    if config.name_dataset == 'fi2010':
                        outputs, _ = model(features)
                    else:
                        outputs, _ = model(features[:, :, 3:])
            if not config.regression:
                outputs = torch.softmax(outputs, dim=-1)

            loss = loss_fn(outputs, targets)

            if config.selection_mode:
                reg_lambda = epoch_i*config.reg_factor/config.anneal if epoch_i<=config.anneal else config.reg_factor
                reg_loss = reg_lambda * reg
                loss += reg_loss

            total_loss.append(loss.item())
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
        # curr_lr = optimizer.param_groups[0]['lr']
        # # print(epoch_i, curr_lr)
        if scheduler and curr_lr > config.min_lr:
            scheduler.step()

        avg_train_loss = np.mean(total_loss)
        if config.selection_mode:
            for name, param in model.named_parameters():
                if name == 'p_logit':
                    sas = torch.sigmoid(param.detach())
                    print(torch.max(sas),torch.min(sas),torch.sum(sas))

                if name == 'feature_select':
                    sas = param.detach()
                    print(torch.max(sas),torch.min(sas),torch.sum(sas))

        # evaluation
        if val_dataloader is not None:

            avg_val_loss, criteria_score, _ = evaluate(model, val_dataloader, device, loss_fn, config)
            # Track the best accuracy
            score = max(criteria_score) + np.mean(np.clip(criteria_score, -0.2, 1)) if config.regression else criteria_score
            if (config.regression and score >= best_score) or (not config.regression and score >= best_score):
                best_score = score

                if config.regression:
                    fig = plt.figure()
                    plt.plot([i for i in range(1, config.forecast_horizon + 1, config.forecast_stride)],
                             np.clip(criteria_score, -50, 1), 'o-', color='g')
                    fig.suptitle(str(epoch_i) + ": " +str(max(criteria_score)))
                    plt.show()

                save_model(model, save_path)
                stop_time = 0
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 4:
                    print(f"Training complete! Best score: {best_score:.5f}%.")
                    print_loss_graph(train_val_loss)
                    return
            print([epoch_i, avg_train_loss, avg_val_loss, np.round(best_score, 5)])
            train_val_loss.append([avg_train_loss, avg_val_loss])

        print("\n")
        print(f"Best score: {best_score:.5f}%.")
    print_loss_graph(train_val_loss)


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
                    outputs, atts = model(features[:,:,3:], inp_mask)
                else:
                    if config.name_dataset == 'fi2010':
                        outputs, _ = model(features)
                    else:
                        outputs, _ = model(features[:,:,3:])

            if not config.regression:
                outputs = torch.softmax(outputs, dim=-1)

            loss = loss_fn(outputs, targets)
                
            val_loss.append(loss.item())

            # Calculate the r2
            if config.regression:
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

        # visualization of attention layer
        # att_all = np.mean(att_all, axis=0)
        # print(att_all.shape)
        # fig, ax = plt.subplots(1, 1, figsize=(24, 10))
        # sns.heatmap(att_all, annot=False, linewidths=0, cmap='gray', robust=True)
        # plt.show()

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
                begin, end = len(pred_all)//2, len(pred_all)//2 + 3000
                forecast_values = np.array(pred_all[begin:end])
                forecast_prob = np.array(prob_all[begin:end])
                positive_prob = np.array(prob_all[begin:end])
                negative_prob = np.array(prob_all[begin:end])
                positive_prob = positive_prob[forecast_values == 2]
                negative_prob = negative_prob[forecast_values == 1]

                buy_throd = np.quantile(positive_prob, config.signal_threshold)
                sell_throd = np.quantile(negative_prob, config.signal_threshold)

                true_values = np.array(lable_all[begin:end])
                prices = np.array(source_all)[begin:end, 0]
                prices = (prices)
                Min, Max = min(prices), max(prices)
                index = np.arange(0, end-begin)

                parameters = {'axes.labelsize': 20,
                              'axes.titlesize': 20,
                              'xtick.labelsize': 18,
                              'ytick.labelsize': 18,
                              'legend.fontsize': 20}
                plt.rcParams.update(parameters)

                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                ax.plot(prices, label='price', alpha=1, color='black')
                # ax.plot(forecast_prob, label='prob', alpha=0.2, color='orange')
                ax.vlines(x=index[np.logical_and(forecast_values == 2,forecast_prob >=buy_throd)], colors='red',
                          ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                ax.vlines(x=index[np.logical_and(forecast_values == 1,forecast_prob >=sell_throd)], colors='green', ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                plt.legend()
                plt.show()

                fig, ax = plt.subplots(1, 1, figsize=(24, 10))
                ax.plot(prices, label='price', alpha=1, color='black')
                ax.vlines(x=index[true_values==2], colors='red',
                          ymin=Min, ymax=Max
                          , linewidths=0.1, alpha=0.6)
                ax.vlines(x=index[true_values==1], colors='green', ymin=Min,
                          ymax=Max
                          , linewidths=0.1, alpha=0.6)
                plt.ylabel('price', fontsize=20)
                plt.legend()
                plt.show()

        backtest_data = None
        if config.backtesting:
            forecast_values = np.array(pred_all)
            forecast_prob = np.array(prob_all)
            positive_prob = np.array(prob_all)
            negative_prob = np.array(prob_all)
            positive_prob = positive_prob[forecast_values == 2]
            negative_prob = negative_prob[forecast_values == 1]
            buy_throd = np.quantile(positive_prob, config.signal_threshold)
            sell_throd = np.quantile(negative_prob, config.signal_threshold)
            short = True

            mid_price = np.array(source_all)[:, 0]
            bid = np.array(source_all)[:, 1]
            ask = np.array(source_all)[:, 2]
            print(bid[0], ask[0])
            trade_delay = config.trade_delay
            trade_fee = config.trade_fee

            DL_machine = tradebot(volume=0.1, fee=trade_fee, short=short)
            Fallow_machine = tradebot(volume=0.1, fee=trade_fee, short=short)
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
            backtest_data = [DL_machine.get_trade_time(),
                             DL_machine.get_winning_rate(),
                             DL_machine.get_underfee_rate(),
                             DL_machine.get_max_drawdown(),
                             DL_his[-1],
                             (mid_price / mid_price[0] - 1)[len(DL_his) - 1],
                             buy_throd,
                             sell_throd]
            print('trade time:', backtest_data[0])
            print('winning rate:', backtest_data[1])
            print('meaningful winning:', backtest_data[2])
            print('max drawback:', backtest_data[3])
            print('highest value:', max(DL_his))
            print('profit:', backtest_data[4])
            # print('sharp: ', DL_machine.get_sharpe())
            print('buy_and_hold:', backtest_data[5])
            print('baseLine profit', FL_his[-1])
            print('baseLine max drawback', Fallow_machine.get_max_drawdown())
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(mid_price/mid_price[0] -1, label='price', alpha=1, color='black')
            ax.plot((np.array(DL_his)), label='profit', alpha=0.7, color='red')
            ax.set_xlim(left=0, right=len(DL_his))
            plt.show()

            # b, w, d = pred_all.shape
            # pred_all = pred_all.reshape(b, w)
            # lable_all = lable_all.reshape(b, w)

    if not config.regression:
        print(classification_report(lable_all, pred_all, zero_division=0, digits=5))
        # print(accuracy_score(lable_all,pred_all))
        return np.mean(val_loss), f1_score(lable_all, pred_all, average='macro'), backtest_data
    else:
        r2 = r2_score(lable_all, pred_all, multioutput='raw_values')
        pred_direction = np.array(pred_all) >= 0
        target_direction = np.array(lable_all) >= 0
        print(classification_report(target_direction, pred_direction, zero_division=0, digits=5))
        print('Task1', np.argmax(r2), max(r2), np.mean(np.clip(r2,-50, 1)))
        return np.mean(val_loss), r2, None


def save_model(model, path, last=False):
    if path is None:
        if last:
            model.save(last=last)
        else:
            model.save()
    else:
        model.save(path)



