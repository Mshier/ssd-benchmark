import numpy as np

def get_labels_start_end_time(frame_wise_labels, bg_class=[]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+1)
    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col)) * 100
    else:
        score = D[-1,-1]
    return score

def edit_score(recog, gt, norm=True, bg_class=[]):
    P, _, _ = get_labels_start_end_time(recog, bg_class)
    Y, _, _ = get_labels_start_end_time(gt, bg_class)
    return levenstein(P, Y, norm)

def f_score(recog, gt, overlap, bg_class=[]):
    p_label, p_start, p_end = get_labels_start_end_time(recog, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(gt, bg_class)
    tp = 0; fp = 0; fn = 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        iou = (1.0*intersection / union) * [p_label[j]==y_label[x] for x in range(len(y_label))]
        idx = np.array(iou).argmax()
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return tp, fp, fn

def evaluate_boundary(pred_bound, gt_bound, tolerance=5):
    tp = 0
    fp = 0
    fn = 0
    gt_pos = np.where(gt_bound == 1)[0]
    pred_pos = np.where(pred_bound == 1)[0]
    used = np.zeros(len(gt_pos), dtype=bool)
    for p in pred_pos:
        matched = False
        for i, g in enumerate(gt_pos):
            if not used[i] and abs(p - g) <= tolerance:
                tp += 1
                used[i] = True
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(gt_pos) - np.sum(used)
    pre = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * pre * rec / (pre + rec + 1e-8)
    return pre*100, rec*100, f1*100

def evaluate_full(recog_list, gt_list, bound_pred_list=None, bound_gt_list=None, tolerance=5, bg_class=[]):
    seg = evaluate_segmentation(recog_list, gt_list, bg_class=bg_class)
    bound = None
    if bound_pred_list is not None and bound_gt_list is not None:
        pre_sum = 0
        rec_sum = 0
        f1_sum = 0
        for p, g in zip(bound_pred_list, bound_gt_list):
            pre, rec, f1 = evaluate_boundary(p, g, tolerance=tolerance)
            pre_sum += pre
            rec_sum += rec
            f1_sum += f1
        bound = {
            "Bound_Prec": pre_sum / len(recog_list),
            "Bound_Rec": rec_sum / len(recog_list),
            "Bound_F1": f1_sum / len(recog_list)
        }
    return seg, bound

