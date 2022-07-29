import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, classification_report, f1_score
import tools

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
    best_r2 = float('-inf')
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

            if config.selection_mode:
                output, reg = model(features)
            else:
                output = model(features)
            if config.name_dataset == 'fi2010':
                output = torch.softmax(output, dim=-1)
            loss = loss_fn(output, targets)

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

            loss.backward()
            total_loss.append(loss.item())

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
        avg_train_loss = np.mean(total_loss)
        if config.selection_mode:
            for name, param in model.named_parameters():
                if name == 'p_logit':
                    sas = torch.sigmoid(param.detach())
                    print(torch.max(sas),torch.min(sas),torch.sum(sas))

        # evaluation
        if val_dataloader is not None:

            avg_val_loss, r2 = evaluate(model, val_dataloader, device, loss_fn, config)

            # Track the best accuracy
            score = max(r2) + np.mean(np.clip(r2,-0.2, 1)) if config.name_dataset =='crypto' else r2
            if score >= best_r2:
                best_r2 = score

                if config.name_dataset == 'crypto':
                    fig = plt.figure()
                    plt.plot([i for i in range(1, config.forecast_horizon + 1, config.forecast_stride)],
                             np.clip(r2, -0.2, 1), 'o-', color='g')
                    fig.suptitle(str(epoch_i) + ": " +str(max(r2)))
                    plt.show()

                torch.save(model.state_dict(), 'best_model_state.bin')
                stop_time = 0
            else:
                stop_time += 1
                print('reach early stop', stop_time)
                if stop_time > 10:
                    print(f"Training complete! Best r2: {best_r2:.2f}%.")
                    tools.print_loss_graph(train_val_loss)
                    return
            print([epoch_i, avg_train_loss, avg_val_loss, r2])
            train_val_loss.append([avg_train_loss, avg_val_loss])
        print("\n")
        print(best_r2)
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
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Load batch to GPU
            features = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Compute logits

            output = model(features)

            # Compute loss
            if config.name_dataset == 'fi2010':
                output = torch.softmax(output, dim=-1)
            loss = loss_fn(output, targets)
            val_loss.append(loss.item())

            # Calculate the r2
            if config.name_dataset =='crypto':
                pred = output.cpu().numpy()
                target = targets.cpu().numpy()
            else:
                pred = torch.argmax(output, dim=1).cpu().numpy()
                target = targets.cpu().numpy()

            pred_all.append(pred)
            lable_all.append(target)

        # Compute the average accuracy and loss over the validation set.
        pred_all = np.concatenate(pred_all, axis=0)
        lable_all = np.concatenate(lable_all, axis=0)
    if config.name_dataset=='fi2010':
        print('Task1', classification_report(lable_all, pred_all, zero_division=0, digits=4))
        return np.mean(val_loss), f1_score(lable_all,pred_all, average='micro')

    else:
        r2 = r2_score(lable_all, pred_all, multioutput='raw_values')
        # r2 = r2_score(lable_all, pred_all)
        print('Task1', np.argmax(r2), max(r2), np.mean(np.clip(r2,-0.2, 1)))
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
#             output = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#
#             # Calculate the accuracy rate
#             logits.extend(F.softmax(output,dim=1).cpu().numpy())
#             preds = torch.argmax(output, dim=1).cpu().numpy()
#             pred_all.extend(preds)
#     return pred_all, logits
