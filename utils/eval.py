import math
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import utils.universal as U


# assignment of gt and detection by score-sorted matchinh, i.e. VOC-style
# predictions should be sorted wrt confidence score
# https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L178
# https://github.com/matterport/Mask_RCNN/issues/326

def assign_gt_det_scoresort(ious, iou_th=0.3):
    if np.any(np.array(ious.shape) == 0):
        return OrderedDict([]), OrderedDict([])

    n_gt = ious.shape[0]
    n_pred = ious.shape[1]

    gt_assign = OrderedDict()
    pred_assign = OrderedDict()

    for i in range(n_pred):
        pred_ious = ious[:,i]
        # find the best matching gt
        sorted_idx = np.argsort(pred_ious)[::-1]
        for j in sorted_idx:
            # matched earlier
            if j in gt_assign.keys():
                continue
            # not matching
            if pred_ious[j] < iou_th:
                continue
            gt_assign.update([(j, [i])])
            pred_assign.update([(i, [j])])
            break

    return gt_assign, pred_assign


# step 1) filter by iou_th -> matching candidates
# step 2) hungarian algo on matching scores -> select best match
def assign_gt_det_hunscore(ious, scores, iou_th=0.3):
    if np.any(np.array(ious.shape) == 0):
        return OrderedDict([]), OrderedDict([])

    match_cand = ious >= iou_th
    costs = np.logical_not(match_cand).astype(scores.dtype) * np.finfo(scores.dtype).max

    n_gt = ious.shape[0]
    n_pred = ious.shape[1]

    for j in range(n_gt):
        for i in range(n_pred):
            if match_cand[j,i]:
                costs[j,i] = 1.0-scores[i]

    costs2 = np.copy(costs) # it might overwrite
    pairs = linear_sum_assignment(costs)

    gt_assign = dict()
    pred_assign = dict()
    for row, col in zip(pairs[0], pairs[1]):
        if (row < costs.shape[0]) and (col < costs.shape[1]) and (costs2[row,col] <= 1.0): # good match
            if col in pred_assign:
                pred_assign[col].append(row)
            else:
                pred_assign[col] = [row]
            if row in gt_assign:
                gt_assign[row].append(col)
            else:
                gt_assign[row] = [col]
    return gt_assign, pred_assign


# assignment of gt and detection, i.e. best matching via Hungarian algorithm
def assign_gt_det_huniou(ious, iou_th=0.3):
    if np.any(np.array(ious.shape) == 0):
        return OrderedDict([]), OrderedDict([])
    min_cost = 1.0-iou_th
    costs = 1.0-ious
    costs2 = np.copy(costs) # it might overwrite
    pairs = linear_sum_assignment(costs)

    gt_assign = dict()
    pred_assign = dict()
    for row, col in zip(pairs[0], pairs[1]):
        if (row < costs.shape[0]) and (col < costs.shape[1]) and (costs2[row,col] <= min_cost): # good match
            if col in pred_assign:
                pred_assign[col].append(row)
            else:
                pred_assign[col] = [row]
            if row in gt_assign:
                gt_assign[row].append(col)
            else:
                gt_assign[row] = [col]
    return gt_assign, pred_assign

def get_area(b, box_format: str):
    """Gets the area of boxes.

    Args:
        b: boxes, size of dimension 1 must be 4.
        box_format: see the description above.

    Returns:
        the area of the boxes: same shape as input, except that dimension 1 is missing.
    """
    if box_format == "xywh":
        return b[:, 2] * b[:, 3]
    elif box_format == "ltrb":
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    else:
        raise ValueError("Invalid box format %s" % (box_format,))


def xywh_to_ltrb(b):
    half_size = b[:, 2:] * 0.5
    mins = b[:, :2] - half_size
    maxes = b[:, :2] + half_size
    return U.concatenate((mins, maxes), 1)


def to_ltrb(b, box_format: str):
    """Converts boxes to ltrb format.

    Args:
        b: boxes, size of dimension 1 must be 4.
        box_format: see the description above.

    Returns:
        the converted boxes: same shape as input.
    """
    if box_format == "ltrb":
        return b
    elif box_format == "xywh":
        return xywh_to_ltrb(b)
    else:
        raise ValueError("Invalid box format %s" % (box_format,))

def box_iou(b1, b2, box_format: str):
    '''Calculates IoUs of boxes.

    Args:
        b1: first box collection, shape=[N,4].
        b2: second box collection, shape=[M,4].
        box_format: "ltrb" or "xywh".

    Returns:
        IoU matrix, shape=[N,M].
    '''
    assert U.ndim(b1) == 2, U.shape(b1)
    b1 = U.expand_dims(b1, 2)  # [N,4,1]
    b1 = to_ltrb(b1, box_format=box_format)

    assert U.ndim(b2) == 2, U.shape(b2)
    b2 = U.transpose(b2, (1, 0))
    b2 = U.expand_dims(b2, 0)  # [1,4,M]
    b2 = to_ltrb(b2, box_format=box_format)

    intersect_mins = U.maximum(b1[:, :2], b2[:, :2])
    intersect_maxes = U.minimum(b1[:, 2:], b2[:, 2:])
    intersect_wh = U.relu(intersect_maxes - intersect_mins)
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
    b1_area = get_area(b1, box_format="ltrb")
    b2_area = get_area(b2, box_format="ltrb")
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def filt_data(annotation, valid_idx):
    keys = list(annotation.keys())
    annotation_filt = {key: annotation[key][valid_idx] for key in keys}
    return annotation_filt


def sort_data(annotation, sort_key='score', desc=False):
    """
    Sorts a dict of lists according to a key.
    """
    keys = list(annotation.keys())
    idx = np.argsort(annotation[sort_key])
    if desc:
        idx = idx[::-1]
    annotation_filt = {key: annotation[key][idx] for key in keys}
    return annotation_filt


def merge_data(annotations):
    """ Merges a list of dicts into a dict of lists. """
    keys = list(annotations[0].keys())
    merged = {key: np.concatenate([annotation[key] for annotation in annotations], axis=0) for key in keys}
    return merged


def calc_recall_precision(pred_corrects, n_gt):
    """
    Computes recall and precision from correct/incorrect detections.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    Predictions should be sorted wrt their score.
    The code is very similar to MaskRCNN implementation https://github.com/matterport/Mask_RCNN/issues/326

    :param pred_corrects: Array of booleans, true if the corresponding prediction has a GT match
    """
    assert pred_corrects.dtype == np.bool

    _cumsum = np.cumsum(pred_corrects).astype(np.float32)
    recs = _cumsum / n_gt
    precs = _cumsum / np.arange(1, len(pred_corrects) + 1)
    return recs, precs


def fix_zigzag(precs):
    """
    Fix the zigzags in precisions,
    see https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    Precs must be sorted wrt recs
    Should be the same as in Pascal VOC https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
    """
    return np.maximum.accumulate(precs[::-1])[::-1]


def calc_ap_auc(_recs, _precs):
    """
    Calculates the ap from precision and recall using the auc, see:
     https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    Adopted Pascal VOC code: https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
    or use from Mask-RCNN: https://github.com/matterport/Mask_RCNN/issues/326
    """
    if len(_recs) == 0:
        return 0.0

    mprec = np.concatenate([[0.0], _precs, [0.0]])
    mrec = np.concatenate([[0.0], _recs, [1.0]])

    precs_fixed = fix_zigzag(mprec)
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * precs_fixed[i + 1])

    return ap


def calc_interp(precs, recs, rec):
    """
    Calculates interpolated precision at a given recall.
    """
    idx = recs >= rec
    if np.sum(idx) > 0:
        pres_inter = np.amax(precs[idx])
    else:
        pres_inter = 0.0
    return pres_inter


def calc_interp_precision(recall, precision, recall_values=np.linspace(start=0.0, stop=1.0, num=11, endpoint=True)):
    """
    Calculates interpolated precision at given recalls (e.g. 11 points linearly spanning the 0..1 range)
    """
    N_rec = len(recall_values)
    precs_inter = []
    for n in range(N_rec):
        rec = recall_values[n]
        prec_inter = calc_interp(precision, recall, rec)
        precs_inter.append(prec_inter)
    return np.array(precs_inter, dtype=precision.dtype)


def calc_ap_interp(recall, precision, recall_values=np.linspace(start=0.0, stop=1.0, num=11, endpoint=True)):
    """
    Calculates the ap from precision and recall using interpolation.
    https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    Should give the same as Pascal VOC code: https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
    Assumes the recall_values are evenly distributed.
    """
    precs_inter = calc_interp_precision(recall, precision, recall_values)
    ap = np.sum(precs_inter) / len(recall_values)
    return ap.astype(np.float32)


def calc_optimal_op_index(recs, precs, scores):
    """
    Determine the index of the optimal operating point. It maximizes precision*recall.
    Input should be sorted wrt scores.
    """
    if len(recs) == 0:
        return -1
    pr = recs * precs
    pr_max = np.amax(pr)
    idx = pr == pr_max
    offs = np.where(idx)[0]
    if len(offs) == 1:
        return offs[0]
    else:
        scores_cand = scores[idx]
        max_idx = np.where((scores_cand == np.amax(scores_cand)))[0][-1]
        min_idx = np.where((scores_cand == np.amin(scores_cand)))[0][0]
        if max_idx == min_idx:
            return offs[min_idx]
        if scores_cand[max_idx] == scores_cand[min_idx]:
            return offs[min_idx]
        return (offs[max_idx], offs[min_idx])


def filter_class(objects, class_field, class_id):
    result = []
    for frame_objects in objects:
        valid_idx = frame_objects[class_field] == class_id
        result.append(filt_data(frame_objects, valid_idx))

    return result


def evaluate_ap(gts, preds, eval_class=None, filt_functions=None, iou_th=0.3,
                iou_func=lambda g, p: box_iou(g, p, 'ltrb'), iou_field='bbox', val_field=None, sim_func=None,
                score_field='score', class_field='type', n_inter=11, assign_method='hunscore', val_metric_name=None, calc_op=True,
                return_curves=True, unlabeled_classes=[], save_matching=None, rotated=False, save_pr=False, pr_name=''):
    """
    Calculates the AP metric for a given IoU threshold. The implementation is based on the Detectron code and this summary:
    https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

    Main steps:
     1) computes GT-prediction IoUs
     2) finds best GT-prediction assignment
     3) determine correct predictions
     4) sorts predictions wrt confidence score
     5) compute recall and precision
     6) fix zigzag in precisions
     7) compute area under recall-precision curve

    It is also possible to calculate the error of an attribute over the matched pairs. The attribute name must be passed
    in `val_field`, `sim_func` calculates the error. The metric name in the results will be `val_metric_name`.

    :param gts: Dictionary of lists, holding the ground truth boxes.
    :param preds: Dictionary of lists, holding the predictions.
    :param eval_class: If given, the metrics are calculated only on the elements of this class
    :param iou_field: the field that holds the bounding boxes
    :param n_inter:  number of interpolation points for Pascal VOC-style AP metrics
    :param assign_method: the matching algorithm
    :param calc_op: calculate optimal detection threshold (maximizes precision*recall)
    :param unlabeled_classes: These classes are ignored *after* association, so GT knowledge is used for filtering.
                            Useful for unlabeled examples.
    :return:
    """
    # Filter to class
    if eval_class is not None:
        gts = filter_class(gts, class_field, eval_class)
        preds = filter_class(preds, class_field, eval_class)

    # Apply other filters
    if filt_functions is not None:
        new_gt = []
        for gt_frame in gts:
            filt_idx = np.ones(len(gt_frame[iou_field]), dtype=np.bool)
            if len(gt_frame[iou_field]) > 0:
                for filt_function in filt_functions:
                    filt_idx = np.bitwise_and(filt_idx, filt_function(gt_frame))
            gt_frame.update([('filter', filt_idx)])
            new_gt.append(gt_frame)
        gts = new_gt

    # Sort predictions by score
    preds = [sort_data(x, sort_key=score_field, desc=True) for x in preds]
    bbox_dims = 5 if rotated else 4
    for pred in preds:
        if len(pred['class']) == 0:
            pred['class'] = pred['class'].reshape(-1)
            pred['bbox'] = pred['bbox'].reshape(0, bbox_dims)
            pred['img_name'] = pred['img_name'].reshape(-1)
            pred['yaw'] = pred['yaw'].reshape(-1)
            pred['score'] = pred['score'].reshape(-1)
    for gt in gts:
        if len(gt['class']) == 0:
            gt['class'] = gt['class'].reshape(-1)
            gt['bbox'] = gt['bbox'].reshape(0, bbox_dims)
            gt['img_name'] = gt['img_name'].reshape(-1)
            gt['yaw'] = gt['yaw'].reshape(-1)
    associations = []

    if len(gts) > 0:
        for gt, pred in zip(gts, preds):
            # step 1 - compute gt-pred iou
            ious = iou_func(gt[iou_field], pred[iou_field])  # compute gt-pred overlaps
            # step 2 - assign preds to gts, using Hungarian algorithm
            if assign_method == 'scoresort':
                gt_assigns, pred_assigns = assign_gt_det_scoresort(ious, iou_th=iou_th)
            elif assign_method == 'huniou':
                gt_assigns, pred_assigns = assign_gt_det_huniou(ious, iou_th=iou_th)
            elif assign_method == 'hunscore':
                gt_assigns, pred_assigns = assign_gt_det_hunscore(ious, pred[score_field], iou_th=iou_th)
            elif assign_method == 'distance':
                gt_assigns, pred_assigns = assign_gt_det_distance(ious, distance_thr=iou_th)
            else:
                raise NotImplementedError(assign_method + ' assignment not implemented')

            pred_assigment = -np.ones(len(pred[iou_field]),
                                      dtype=np.int32)  # shape: (nPreds,); pred_assigment[i] is the pair in gts
            for pred_key, pred_assign in pred_assigns.items():
                if len(pred_assign) == 1:
                    pred_assigment[pred_key] = pred_assign[0]
                elif len(pred_assign) > 1:
                    print('multiple matches found, keeping the best match')
                    gt_ious = ious[:, pred_key]
                    gt_overlaps = gt_ious[pred_assign]
                    best_idx = np.argmax(gt_overlaps)
                    pred_assigment[pred_key] = pred_assign[best_idx]
                else:
                    continue

            correct = pred_assigment != -1  # correct prediction = assigned to a gt
            pred['correct'] = correct

            if val_field is not None:
                pred_assigment_vals = np.zeros(len(pred[iou_field]), dtype=gt[val_field].dtype)
                pred_assigment_vals[correct] = gt[val_field][pred_assigment[correct]]
                pred.update([('assigned_' + val_field, pred_assigment_vals)])

            gt_assigment = -np.ones(len(gt[iou_field]), dtype=np.int32)
            gt_assigment[pred_assigment[correct]] = np.where(correct)[0]
            detected = gt_assigment != -1
            gt['detected'] = detected

            if save_matching:
                assoc = {'pred_' + k: v for k, v in pred.items()}
                for k in gt:
                    new_shape = list(gt[k].shape)
                    new_shape[0] = len(pred[iou_field])
                    assoc['gt_' + k] = np.zeros(new_shape, dtype=gt[k].dtype)
                    assoc['gt_' + k][correct] = gt[k][pred_assigment[correct]]
                assoc['pred_valid'] = np.ones(len(pred[iou_field]), dtype='bool')
                assoc['gt_valid'] = assoc['gt_detected']
                associations.append(assoc)

                # add false negatives
                assoc = {'gt_' + k: v[~gt['detected']] for k, v in gt.items()}
                num_undetected = np.sum(~gt['detected'])
                for k in pred:
                    new_shape = list(pred[k].shape)
                    new_shape[0] = num_undetected
                    assoc['pred_' + k] = np.zeros(new_shape, dtype=pred[k].dtype)
                assoc['pred_valid'] = np.zeros(num_undetected, dtype='bool')
                assoc['gt_valid'] = np.ones(num_undetected, dtype='bool')
                associations.append(assoc)

            # Filter out ignored classes
            if len(unlabeled_classes) > 0:
                ignore_gt = np.isin(gt[class_field], unlabeled_classes)
                ignore_pred = np.zeros_like(pred['correct'], dtype='bool')
                ignore_pred[gt_assigment[ignore_gt & gt['detected']]] = True
                ignore_pred = ignore_pred | np.isin(pred[class_field], unlabeled_classes)

                for k in gt:
                    gt[k] = gt[k][~ignore_gt]
                for k in pred:
                    pred[k] = pred[k][~ignore_pred]

        gts = merge_data(gts)
        preds = merge_data(preds)
        if save_matching:
            associations = merge_data(associations)

        # step 3 - sort predictions wrt score
        preds = sort_data(preds, sort_key=score_field, desc=True)
        recs, precs = calc_recall_precision(preds['correct'], n_gt=len(gts[iou_field]))

        if len(recs) > 0:
            recall = recs[-1]
            precision = precs[-1]
            if filt_functions is not None:
                # normalize recall by filter
                # TODO: check this
                recs = recs * (len(gts['filter']) / np.sum(gts['filter']))
                recs = np.clip(recs, a_min=0, a_max=1)

            recs_fixed = np.concatenate([[0.0], recs, [1.0]])
            precs_fixed = fix_zigzag(np.concatenate([[0.0], precs, [0.0]]))

            if save_pr:
                plot1 = plt.figure(1)
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.plot(recs_fixed, precs_fixed)
                plt.savefig(pr_name)

                plt.figure().clear()
                plt.close()
                plt.cla()
                plt.clf()

                n_bins = 10
                plot2 = plt.figure(2)
                fig, axs = plt.subplots(1, 1, tight_layout=True)
                axs.hist(preds['score'], bins=n_bins)
                plt.savefig('hist_' + pr_name)

            ap_auc = calc_ap_auc(recs, precs)
            recall_values = np.linspace(start=0.0, stop=1.0, num=n_inter, endpoint=True)
            prec_interp = calc_interp_precision(recs, precs, recall_values=recall_values)
            ap_interp = calc_ap_interp(recs, precs, recall_values=recall_values)

            if val_field is not None:
                similarity = sim_func(preds[val_field], preds['assigned_' + val_field])
                avg_val_metric = np.mean(similarity[preds['correct']])

            if calc_op:
                if np.isnan(recall):
                    op_idx = 0
                else:
                    op_idx = calc_optimal_op_index(recs, precs, preds[score_field])
                if isinstance(op_idx, (tuple, list)):
                    # TBD: if anyone has better idea
                    # e.g. use op_idx[1] only -> favour smaller scores and more recall, but lower precision
                    # e.g. use op_idx[0] only -> favour higher scores and more precise detections, but less detections overall
                    rec_opt = math.sqrt(recs[op_idx[0]] * recs[op_idx[1]])
                    prec_opt = math.sqrt(precs[op_idx[0]] * precs[op_idx[1]])
                    score_opt = math.sqrt(preds[score_field][op_idx[0]] * preds[score_field][op_idx[1]])
                else:
                    rec_opt = recs[op_idx]
                    prec_opt = precs[op_idx]
                    score_opt = preds[score_field][op_idx]
        else:
            ap_auc = 0.0
            ap_interp = 0.0
            avg_val_metric = 0.0
            recall = 0.0
            precision = 0.0
            if calc_op:
                rec_opt = 0.0
                prec_opt = 0.0
                score_opt = 0.0
            recs = np.empty((0,), dtype=np.float32)
            precs = np.empty((0,), dtype=np.float32)
            recs_fixed = np.empty((0,), dtype=np.float32)
            precs_fixed = np.empty((0,), dtype=np.float32)
            recall_values = np.empty((0,), dtype=np.float32)
            prec_interp = np.empty((0,), dtype=np.float32)
    else:
        ap_auc = 0.0
        ap_interp = 0.0
        avg_val_metric = 0.0
        recall = 0.0
        precision = 0.0
        if calc_op:
            rec_opt = 0.0
            prec_opt = 0.0
            score_opt = 0.0
        recs = np.empty((0,), dtype=np.float32)
        precs = np.empty((0,), dtype=np.float32)
        recs_fixed = np.empty((0,), dtype=np.float32)
        precs_fixed = np.empty((0,), dtype=np.float32)
        recall_values = np.empty((0,), dtype=np.float32)
        prec_interp = np.empty((0,), dtype=np.float32)

    if save_matching:
        assert np.sum(associations['pred_correct']) == np.sum(associations['gt_detected'])
        assert np.sum(associations['pred_correct']) == 0 or \
               np.all(associations['pred_img_name'][associations['pred_correct']] ==
                      associations['gt_img_name'][associations['gt_detected']])

        del associations['pred_correct']
        del associations['gt_detected']
        with open(save_matching, 'wb') as f:
            pickle.dump(associations, f)

    res = {
        'ap_auc': ap_auc,
        'ap_interp': ap_interp,
        'recall_2d': recall,
        'precision_2d': precision
    }
    if val_field is not None:
        res[val_metric_name] = avg_val_metric
    if calc_op:
        res.update({
            'recall_op': rec_opt,
            'precision_op': prec_opt,
            'score_op': score_opt
        })
    if return_curves:
        res.update({
            'recall', recs,
            'precision', precs,
            'recall_fixed', recs_fixed,
            'precision_fixed', precs_fixed,
            'recall_interp', recall_values,
            'precision_interp', prec_interp
        })

    return res


def obj2arrays(objects, needs_score=False):
    """
    Converts a list of dicts describing objects to a dict of arrays.
    The output will have the field 'bbox' having the 2D bbox in ltrb format and
    the objectness scores if needs_score is True.
    """
    bboxes = []
    class_name = []
    img_name = []
    for obj in objects:
        bboxes.append([obj['BoundingBox2D X1'], obj['BoundingBox2D Y1'],
                       obj['BoundingBox2D X2'], obj['BoundingBox2D Y2']])
        class_name.append(obj['ObjectType'])
        img_name.append(obj.get('img_name', None))

    bboxes = np.array(bboxes)
    class_name = np.array(class_name)
    img_name = np.array(img_name)
    if len(bboxes) == 0:
        bboxes = bboxes.reshape(0, 4)

    result = {'class': class_name, 'bbox': bboxes, 'img_name': img_name}

    if needs_score:
        result['score'] = np.array([x['Detection Score'] for x in objects])

    return result


def obj2arrays_bev(objects, classes, scores=None, name=None, needs_score=False, x_range=200, rotated=False):
    """
    Converts a list of dicts describing objects to a dict of arrays.
    The output will have the field 'bbox' having the 2D bbox in ltrb format and
    the objectness scores if needs_score is True.
    """
    bboxes = []
    class_name = np.array(classes)
    img_name = []
    yaw = []

    for obj in objects:
        x, y = obj[0], obj[1]
        l, w = obj[3], obj[4]
        alpha = np.degrees(obj[6])
        if rotated:
            bboxes.append([x, y, l, w, alpha])
        else:
            x1, x2 = x - l/2, x + l/2
            y1, y2 = y - w/2, y + w/2
            bboxes.append([x1, y1, x2, y2])

        yaw.append(alpha)
        img_name.append(name)

    bboxes = np.array(bboxes)
    img_name = np.array(img_name)
    yaw = np.array(yaw)
    if len(bboxes) == 0:
        bboxes = bboxes.reshape(0, 4)

    valids = np.abs(objects[:, 0]) <= x_range
    result = {'class': class_name[valids], 'bbox': bboxes[valids], 'img_name': img_name[valids], 'yaw': yaw[valids]}

    if needs_score:
        result['score'] = np.array(scores)[valids]

    return result


class MAPCalculator:
    def __init__(self):
        self.preds = []
        self.gts = []
        self.img_names = []
        self.output_name = os.path.join('', 'map_association.pkl')
        self.ignore_classes = []

    def __len__(self):
        return len(self.preds)

    def reset(self):
        self.preds = []
        self.gts = []
        self.img_names = []

    def update(self, gt, pred, img_names=None):
        """
        gt and pred are list of list of detections (frame, then detections)
        """
        # assert len(gt) == len(pred)
        # if img_names:
        #     gt = deepcopy(gt)
        #     for i, frame_gt in enumerate(gt):
        #         for box in frame_gt:
        #             box['img_name'] = img_names[i]
        #
        #     pred = deepcopy(pred)
        #     for i, frame_pred in enumerate(pred):
        #         for box in frame_pred:
        #             box['img_name'] = img_names[i]

        self.gts.extend(gt)
        self.preds.extend(pred)
        self.img_names.extend(img_names)

    def compute_bev(self, eval_class=None, iou_thr=0.3, assign_method='hunscore', x_range=200,
                    iou_func=lambda g, p: box_iou(g, p, 'ltrb'), rotated=False, save_pr_curve=False, pr_name=''):
        gts = [obj2arrays_bev(objects=x[0], classes=x[1], name=img_name, x_range=x_range, rotated=rotated) for x, img_name in zip(self.gts, self.img_names)]
        preds = [obj2arrays_bev(x[0], x[2], x[1], name=img_name, needs_score=True, x_range=x_range, rotated=rotated) for x, img_name in zip(self.preds, self.img_names)]

        num_preds = sum([len(x['bbox']) for x in preds])

        result = evaluate_ap(gts, preds, iou_th=iou_thr, return_curves=False, val_field='class',
                             sim_func=lambda a, b: a == b, val_metric_name='cls_accuracy_bev',
                             unlabeled_classes=self.ignore_classes, class_field='class', save_matching=None,
                             eval_class=eval_class, assign_method=assign_method, iou_func=iou_func, rotated=rotated,
                             save_pr=save_pr_curve, pr_name=pr_name)

        result['num_preds'] = num_preds

        yaw_result = evaluate_ap(gts, preds, iou_th=iou_thr, return_curves=False, val_field='yaw',
                             sim_func=cosine_similarity, val_metric_name='aos',
                             unlabeled_classes=self.ignore_classes, class_field='class', save_matching=None,
                             eval_class=eval_class, assign_method=assign_method, iou_func=iou_func, rotated=rotated)
        result['aos'] = yaw_result['aos']

        return result


def distance_angle_unsigned(a1, a2):
    diff = np.fmod(np.abs(a1 - a2), 360.0).astype(a1.dtype)
    diff[diff > 180] = 360 - diff[diff > 180]
    return diff


# cosine similarity
# as eqs. 4-5 in http://www.cvlibs.net/publications/Geiger2012CVPR.pdf
def cosine_similarity(v1, v2):
    v1_angle = (v1.astype(np.float64))
    v2_angle = (v2.astype(np.float64))
    diff_angle = distance_angle_unsigned(v1_angle, v2_angle)
    diff_yaw = np.radians(diff_angle)
    sims = (1.0 + np.cos(diff_yaw)) / 2.0
    return sims


def cost_matrix_by_dist(gt_bboxes, pred_bboxes, distance_thr, format='ltrb'):
    """
    gt_bboxes shape: [N, 4] or [N, 5]. ltrb: [x1, y1, x2, y2, {yaw}], xywh: [xc, yc, w, h, {yaw}]
    """
    import scipy
    if format == 'ltrb':
        l = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        w = gt_bboxes[:, 3] - gt_bboxes[:, 1]

        gt_xs = gt_bboxes[:, 0] + l/2
        gt_ys = gt_bboxes[:, 1] + w/2
        gt_centers = np.concatenate([gt_xs[np.newaxis], gt_ys[np.newaxis]], axis=0)

        l = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        w = pred_bboxes[:, 3] - pred_bboxes[:, 1]

        pred_xs = pred_bboxes[:, 0] + l / 2
        pred_ys = pred_bboxes[:, 1] + w / 2
        pred_centers = np.concatenate([pred_xs[np.newaxis], pred_ys[np.newaxis]], axis=0)

    C = scipy.spatial.distance.cdist(gt_centers.T, pred_centers.T)
    C[C > distance_thr] = 999999  # gating

    return C


def assign_gt_det_distance(C, distance_thr=2.):
    rows, cols = linear_sum_assignment(C)
    dist = C[rows, cols]

    # Keep only good matches
    rows = rows[dist < distance_thr]
    cols = cols[dist < distance_thr]

    gt_assign = dict()
    pred_assign = dict()

    for i, row in enumerate(rows):
        gt_assign[row] = [cols[i]]
        pred_assign[cols[i]] = [row]

    return gt_assign, pred_assign
