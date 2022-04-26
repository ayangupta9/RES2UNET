########train.py


# ? https://github.com/ojedaa/Res2Unet/blob/main/metrics2.py

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from SaveBestModel import SaveBestModel, save_model
from new_res2_unet import res2net50_v1b_26w_4s
from tqdm import tqdm
import cv2
import scipy.ndimage as ndimage
from MetricMonitor import MetricMonitor

# from torchmetrics import Accuracy, JaccardIndex, Precision, Recall, F1Score
# import seaborn as sns


def RobertsAlogrithm(image):
    edges = image - ndimage.morphology.binary_erosion(image)
    return edges


def focal_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + 1e-10) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-10) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def boundary_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """

    pred_num = pred.cpu()
    pred_edge = RobertsAlogrithm(pred_num.detach().numpy())
    gt_edge = RobertsAlogrithm(gt.cpu().detach().numpy())

    sum1 = (pred_edge - gt_edge) * (pred_edge - gt_edge)
    sum2 = pred_edge * pred_edge
    sum3 = gt_edge * gt_edge

    return sum1.sum() / (sum2.sum() + sum3.sum() + 1e-10)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    # lr = lr * (0.1 ** (epoch // 30))
    lr = lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def calculate_accuracy(output, target):
    output = output >= 0.5
    target = target == 1.0
    # print(np.unique((target == output).sum(dim=0).cpu().numpy()))
    # print(output.size(0))
    # return torch.true_divide((target == output).sum(), output.size(0)).item()
    return (output == target).sum()


# def train(
#     train_dataloader,
#     eval_dataloader,
#     show_visuals=False,
#     epo_num=6,
# ):

#     torch.cuda.set_device(0)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = res2net50_v1b_26w_4s(pretrained=True)
#     model = model.to(device)
#     criterion = nn.BCELoss().to(device)
#     # optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     optimizer = torch.optim.Adam(
#         params=model.parameters(), lr=0.001, betas=(0.9, 0.999)
#     )
#     save_best_model = SaveBestModel()

#     # optimizer = torch.optim.SGD(
#     #     model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
#     # )

#     all_train_iter_loss = []
#     all_test_iter_loss = []
#     prediction = 0
#     label = 0

#     # start timing
#     print("start training....")
#     prev_time = datetime.now()
#     for epo in range(1, epo_num + 1):

#         metric_monitor = MetricMonitor()
#         train_loss = 0
#         model.train()
#         adjust_learning_rate(optimizer, epo, 0.001)

#         stream = tqdm(train_dataloader)

#         # for index, (ls, ls_msk) in tqdm(enumerate(train_dataloader)):
#         for index, (ls, ls_msk) in enumerate(stream):
#             ls = ls.to(device)
#             ls_msk = ls_msk.to(device)

#             optimizer.zero_grad()
#             output = model(ls)
#             output = torch.sigmoid(output)

#             if index == 0:
#                 prediction = output
#                 label = ls_msk

#             boundary_l = boundary_loss(output, ls_msk)
#             loss = 2 * criterion(output, ls_msk) + boundary_l
#             # loss = criterion(output, ls_msk)
#             metric_monitor.update("Loss", loss.item())
#             loss.backward()

#             iter_loss = loss.item()
#             all_train_iter_loss.append(iter_loss)
#             train_loss += iter_loss

#             stream.set_description(
#                 "Epoch: {epoch}. Train.      {metric_monitor}".format(
#                     epoch=epo, metric_monitor=metric_monitor
#                 )
#             )

#             optimizer.step()

#         test_loss = 0
#         metric_monitor = MetricMonitor()
#         model.eval()
#         stream = tqdm(eval_dataloader)

#         with torch.no_grad():
#             # for index, (ls, ls_msk) in tqdm(enumerate(eval_dataloader)):

#             for index, (ls, ls_msk) in enumerate(stream):
#                 ls = ls.to(device)
#                 ls_msk = ls_msk.to(device)

#                 optimizer.zero_grad()
#                 output = model(ls)
#                 output = torch.sigmoid(
#                     output
#                 )  # output.shape is torch.Size([4, 2, 160, 160])
#                 boundary_l = boundary_loss(output, ls_msk)
#                 loss = 2 * criterion(output, ls_msk) + boundary_l
#                 # loss = criterion(output, ls_msk)
#                 metric_monitor.update("Loss", loss.item())

#                 iter_loss = loss.item()
#                 all_test_iter_loss.append(iter_loss)
#                 test_loss += iter_loss
#                 stream.set_description(
#                     "Epoch: {epoch}. Eval.      {metric_monitor}".format(
#                         epoch=epo, metric_monitor=metric_monitor
#                     )
#                 )

#         cur_time = datetime.now()
#         h, remainder = divmod((cur_time - prev_time).seconds, 3600)
#         m, s = divmod(remainder, 60)
#         time_str = "Time %02d:%02d:%02d" % (h, m, s)
#         prev_time = cur_time

#         print(
#             "epoch train loss = %f, epoch test loss = %f, %s"
#             % (
#                 train_loss / len(train_dataloader),
#                 test_loss / len(eval_dataloader),
#                 time_str,
#             )
#         )

#         save_best_model(
#             current_valid_loss=test_loss / len(eval_dataloader),
#             epoch=epo,
#             model=model,
#             optimizer=optimizer,
#             criterion=criterion,
#         )

#         if show_visuals == True:
#             with torch.no_grad():
#                 plt.figure(figsize=(20, 5))
#                 plt.subplot(1, 4, 1)
#                 plt.imshow(prediction[0].permute(1, 2, 0).cpu().numpy()[:, :, 0] >= 0.5)

#                 plt.subplot(1, 4, 2)
#                 plt.imshow(label[0].permute(1, 2, 0).cpu().numpy()[:, :, 0])
#                 plt.show()

#                 print(prediction.shape, label.shape)

#     save_model(epochs=epo_num, criterion=criterion, model=model, optimizer=optimizer)
#     return model


def train(train_dataloader, eval_dataloader, show_visuals=False, epo_num=100):

    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = res2net50_v1b_26w_4s(pretrained=False)
    model = model.to(device)
    criterion = nn.BCELoss().to(device)
    save_best_model = SaveBestModel()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )

    all_train_iter_loss = []
    all_test_iter_loss = []
    prediction = 0
    label = 0

    # start timing
    print("start training....")
    prev_time = datetime.now()
    for epo in range(1, epo_num + 1):

        train_loss = 0
        model.train()
        adjust_learning_rate(optimizer, epo, 0.001)
        for index, (ls, ls_msk) in tqdm(enumerate(train_dataloader)):
            ls = ls.to(device)
            ls_msk = ls_msk.to(device)

            optimizer.zero_grad()
            output = model(ls)
            output = torch.sigmoid(output)

            if index == 0:
                prediction = output
                label = ls_msk

            # boundary_l = boundary_loss(output, ls_msk)
            # loss = 2 * criterion(output, ls_msk) + boundary_l
            loss = criterion(output, ls_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss

            optimizer.step()

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for index, (ls, ls_msk) in tqdm(enumerate(eval_dataloader)):

                ls = ls.to(device)
                ls_msk = ls_msk.to(device)

                optimizer.zero_grad()
                output = model(ls)
                output = torch.sigmoid(
                    output
                )  # output.shape is torch.Size([4, 2, 160, 160])
                # boundary_l = boundary_loss(output, ls_msk)
                # loss = 2 * criterion(output, ls_msk) + boundary_l
                loss = criterion(output, ls_msk)

                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        if show_visuals == True:
            with torch.no_grad():
                plt.figure(figsize=(20, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(prediction[0].permute(1, 2, 0).cpu().numpy()[:, :, 0] >= 0.5)

                plt.subplot(1, 4, 2)
                plt.imshow(label[0].permute(1, 2, 0).cpu().numpy()[:, :, 0])
                plt.show()

                print(prediction.shape, label.shape)

        print(
            "epoch: %f train loss = %f, epoch test loss = %f, %s"
            % (
                epo,
                train_loss / len(train_dataloader),
                test_loss / len(eval_dataloader),
                time_str,
            )
        )

        save_best_model(
            current_valid_loss=test_loss / len(eval_dataloader),
            epoch=epo,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )
    save_model(epochs=epo_num, model=model, optimizer=optimizer, criterion=criterion)
    model.eval()
    return model


def pretrained_train(
    train_dataloader,
    eval_dataloader,
    pretrained_dict,
    last_eval_loss,
    show_visuals=False,
    epo_num=0,
):
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = res2net50_v1b_26w_4s(pretrained=False)
    model = model.to(device)
    net_dict = model.state_dict()
    model_pretrained_dict = {
        k: v for k, v in pretrained_dict["model_state_dict"].items() if (k in net_dict)
    }
    net_dict.update(model_pretrained_dict)
    model.load_state_dict(net_dict)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999)
    )
    optimizer_net_dict = optimizer.state_dict()
    optimizer_pretrained_dict = {
        k: v
        for k, v in pretrained_dict["optimizer_state_dict"].items()
        if (k in net_dict)
    }
    optimizer_net_dict.update(optimizer_pretrained_dict)
    optimizer.load_state_dict(optimizer_net_dict)

    save_best_model = SaveBestModel(best_valid_loss=last_eval_loss)

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    # )

    all_train_iter_loss = []
    all_test_iter_loss = []
    prediction = 0
    label = 0

    # start timing
    print("start training....")
    prev_time = datetime.now()

    for epo in range(1, epo_num + 1):
        train_loss = 0
        model.train()
        adjust_learning_rate(optimizer, epo, 0.001)
        for index, (ls, ls_msk) in tqdm(enumerate(train_dataloader)):
            ls = ls.to(device)
            ls_msk = ls_msk.to(device)

            optimizer.zero_grad()
            output = model(ls)
            output = torch.sigmoid(output)

            if index == 0:
                prediction = output
                label = ls_msk

            # boundary_l = boundary_loss(output, ls_msk)
            # loss = 2 * criterion(output, ls_msk) + boundary_l
            loss = criterion(output, ls_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss

            optimizer.step()

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for index, (ls, ls_msk) in tqdm(enumerate(eval_dataloader)):

                ls = ls.to(device)
                ls_msk = ls_msk.to(device)

                optimizer.zero_grad()
                output = model(ls)
                output = torch.sigmoid(
                    output
                )  # output.shape is torch.Size([4, 2, 160, 160])
                # boundary_l = boundary_loss(output, ls_msk)
                # loss = 2 * criterion(output, ls_msk) + boundary_l
                loss = criterion(output, ls_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        if show_visuals == True:
            with torch.no_grad():
                plt.figure(figsize=(20, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(prediction[0].permute(1, 2, 0).cpu().numpy()[:, :, 0] >= 0.5)

                plt.subplot(1, 4, 2)
                plt.imshow(label[0].permute(1, 2, 0).cpu().numpy()[:, :, 0])
                plt.show()

                print(prediction.shape, label.shape)

        print(
            "epoch: %f train loss = %f, epoch test loss = %f, %s"
            % (
                epo,
                train_loss / len(train_dataloader),
                test_loss / len(eval_dataloader),
                time_str,
            )
        )

        save_best_model(
            current_valid_loss=test_loss / len(eval_dataloader),
            epoch=epo,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

    save_model(epochs=epo_num, model=model, optimizer=optimizer, criterion=criterion)
    model.eval()
    return model


"""
    # for epo in range(1, epo_num + 1):
    #     metric_monitor = MetricMonitor()
    #     train_loss = 0
    #     model.train()
    #     adjust_learning_rate(optimizer, epo, 0.001)

    #     stream = tqdm(train_dataloader)

    #     # for index, (ls, ls_msk) in tqdm(enumerate(train_dataloader)):
    #     for index, (ls, ls_msk) in enumerate(stream):
    #         ls = ls.to(device)
    #         ls_msk = ls_msk.to(device)

    #         optimizer.zero_grad()
    #         output = model(ls)
    #         output = torch.sigmoid(output)

    #         if index == 0:
    #             prediction = output
    #             label = ls_msk

    #         boundary_l = boundary_loss(output, ls_msk)
    #         loss = 2 * criterion(output, ls_msk) + boundary_l
    #         # loss = criterion(output, ls_msk)
    #         metric_monitor.update("Loss", loss.item())
    #         loss.backward()

    #         iter_loss = loss.item()
    #         all_train_iter_loss.append(iter_loss)
    #         train_loss += iter_loss

    #         stream.set_description(
    #             "Epoch: {epoch}. Train.      {metric_monitor}".format(
    #                 epoch=epo, metric_monitor=metric_monitor
    #             )
    #         )

    #         optimizer.step()

    #     test_loss = 0
    #     metric_monitor = MetricMonitor()
    #     model.eval()
    #     stream = tqdm(eval_dataloader)

    #     with torch.no_grad():
    #         # for index, (ls, ls_msk) in tqdm(enumerate(eval_dataloader)):

    #         for index, (ls, ls_msk) in enumerate(stream):
    #             ls = ls.to(device)
    #             ls_msk = ls_msk.to(device)

    #             optimizer.zero_grad()
    #             output = model(ls)
    #             output = torch.sigmoid(
    #                 output
    #             )  # output.shape is torch.Size([4, 2, 160, 160])
    #             boundary_l = boundary_loss(output, ls_msk)
    #             loss = 2 * criterion(output, ls_msk) + boundary_l
    #             # loss = criterion(output, ls_msk)
    #             metric_monitor.update("Loss", loss.item())

    #             iter_loss = loss.item()
    #             all_test_iter_loss.append(iter_loss)
    #             test_loss += iter_loss
    #             stream.set_description(
    #                 "Epoch: {epoch}. Eval.      {metric_monitor}".format(
    #                     epoch=epo, metric_monitor=metric_monitor
    #                 )
    #             )

    #     cur_time = datetime.now()
    #     h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    #     m, s = divmod(remainder, 60)
    #     time_str = "Time %02d:%02d:%02d" % (h, m, s)
    #     prev_time = cur_time

    #     print(
    #         "epoch train loss = %f, epoch test loss = %f, %s"
    #         % (
    #             train_loss / len(train_dataloader),
    #             test_loss / len(eval_dataloader),
    #             time_str,
    #         )
    #     )

    #     save_best_model(
    #         current_valid_loss=test_loss / len(eval_dataloader),
    #         epoch=epo,
    #         model=model,
    #         optimizer=optimizer,
    #         criterion=criterion,
    #     )

    #     if show_visuals == True:
    #         with torch.no_grad():
    #             plt.figure(figsize=(20, 5))
    #             plt.subplot(1, 4, 1)
    #             plt.imshow(prediction[0].permute(1, 2, 0).cpu().numpy()[:, :, 0] >= 0.5)

    #             plt.subplot(1, 4, 2)
    #             plt.imshow(label[0].permute(1, 2, 0).cpu().numpy()[:, :, 0])
    #             plt.show()

    #             print(prediction.shape, label.shape)

    save_model(epochs=epo_num, criterion=criterion, model=model, optimizer=optimizer)
    return model
"""

# if __name__ == "__main__":
#     train(epo_num=50)
