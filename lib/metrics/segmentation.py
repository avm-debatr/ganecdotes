import torch
import numpy as np
import skimage.measure as measure
import skimage.segmentation as seg
import skimage.metrics as metrics
import matplotlib.pyplot as plt


def get_mask_iou(gt_mask, pred_mask, label):

    gt, pred = np.zeros_like(gt_mask), np.zeros_like(pred_mask)

    gt[gt_mask==label] = 1
    pred[pred_mask==label] = 1

    gt[gt_mask!=label] = 0
    pred[pred_mask!=label] = 0

    intersec = gt*pred
    union    = gt + pred
    # union[union>0] = 0
    # intersec = gt_mask*pred_mask
    # union = gt_mask + pred_mask
    # union[union>1] = 1

    if np.count_nonzero(union) > 0:
        return np.count_nonzero(intersec)/np.count_nonzero(union)
    else:
        return 0


def get_bb_iou(gt_mask, pred_mask, label):
    gt, pred = np.zeros_like(gt_mask), np.zeros_like(pred_mask)

    gt[gt_mask==label] = 1
    pred[pred_mask==label] = 1

    gt[gt_mask!=label] = 0
    pred[pred_mask!=label] = 0

    nz = gt.nonzero()

    if gt.max()==0:
        return 0

    xmin, xmax, ymin, ymax = nz[0].min(), nz[0].max(), \
                             nz[1].min(), nz[1].max()

    gt[xmin:xmax, ymin:ymax] = 1

    nz = pred.nonzero()

    if pred.max()==0:
        return 0

    xmin, xmax, ymin, ymax = nz[0].min(), nz[0].max(), \
                             nz[1].min(), nz[1].max()

    pred[xmin:xmax, ymin:ymax] = 1

    intersec = gt * pred
    union = gt + pred
    # union[union > 0] = 1

    if np.count_nonzero(union) > 0:
        return np.count_nonzero(intersec) / np.count_nonzero(union)
    else:
        return 0


def get_mask_dice(gt_mask, pred_mask, label):
    iou = get_mask_iou(gt_mask, pred_mask, label)
    return 2*iou/(iou + 1)


def get_bb_dice(gt_mask, pred_mask, label):
    iou = get_bb_iou(gt_mask, pred_mask, label)
    return 2*iou/(iou + 1)


def get_roc_curve(iou_scores, iou_thresh = 0.5):
    return


def plot_roc_curve(roc_curves, fp):
    return


def get_mean_avg_precision(iou_scores):
    return


def get_weighted_iou(gt_mask, mask_iou, classes):

    w_iou = 0

    h, w, = gt_mask.shape
    gt_area = h*w

    for i, c in enumerate(classes):

        if c!='background':
            wt = np.count_nonzero(gt_mask==i)
            wt /= gt_area
            w_iou += wt*mask_iou[c]

    return w_iou


def get_bin_iou(gt_mask, pred_mask):

    gt_fg_mask, pred_fg_mask = gt_mask.copy(), \
                               pred_mask.copy()

    gt_fg_mask[gt_fg_mask>0] = 1
    pred_fg_mask[pred_fg_mask>0] = 1

    intersec = gt_fg_mask*pred_fg_mask
    union    = gt_fg_mask + pred_fg_mask

    if np.count_nonzero(union) > 0:
        return np.count_nonzero(intersec)/np.count_nonzero(union)
    else:
        return 0


def get_pd_at_iou_threshold(iou_scores,
                            classes,
                            iou_thr=0.5):

    pd_scores = {c: (iou_scores[c]>iou_thr).mean()
                 for c in classes}

    return pd_scores


def get_iou_vs_pd_curve(iou_pd,
                        classes,
                        iou_inc=0.05):

    num_vals = int(1/iou_inc)
    vals = np.linspace(0, 1, num_vals)

    iou_vs_pd_curve = [get_pd_at_iou_threshold(iou_pd,
                                               classes,
                                               t)
                       for t in vals]

    iou_vs_pd_dict = {c:[] for c in classes}

    for iou_vs_pd in iou_vs_pd_curve:
        for c in classes:
            iou_vs_pd_dict[c].append(iou_vs_pd[c])

    mean_curve = np.zeros_like(vals)

    for k, v in iou_vs_pd_dict.items():
        mean_curve += v

    mean_curve /= len(classes)
    iou_vs_pd_dict['Mean'] = mean_curve

    return iou_vs_pd_dict


def plot_iou_vs_pd_curve(curves,
                         classes,
                         fname,
                         expt_name):

    plt.figure()
    x_val = np.linspace(0, 1, len(curves[classes[0]]))

    # mean_curve = np.zeros_like(x_val)

    for i, c in enumerate(classes):
        if c=='Mean':
            plt.plot(x_val, curves[c], label=c, color='black')
        else:
            plt.plot(x_val, curves[c], ':', label=c)

        # mean_curve += curves[c]

    # mean_curve /= (i+1)

    plt.ylim([0, 1.2])
    plt.xlim([0, 1])
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlabel('IoU Threshold')
    plt.ylabel('PD')
    plt.title(f"IoU vs PD Curve, Test Class - {expt_name}")

    plt.savefig(fname)
    plt.close()

