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

def evaluate_segmentation(recog_list, gt_list, bg_class=[]):
    total_correct = 0
    total_frames = 0
    total_edit = 0
    overlaps = [0.1, 0.25, 0.5]
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    for recog, gt in zip(recog_list, gt_list):
        total_frames += len(gt)
        total_correct += np.sum(np.array(recog) == np.array(gt))
        total_edit += edit_score(recog, gt, bg_class=bg_class)
        for s, th in enumerate(overlaps):
            t, f, n = f_score(recog, gt, th, bg_class=bg_class)
            tp[s] += t
            fp[s] += f
            fn[s] += n

    acc = 100.0 * total_correct / total_frames
    edit = total_edit / len(recog_list)
    f1s = []
    for s in range(3):
        pre = tp[s] / (tp[s] + fp[s] + 1e-8)
        rec = tp[s] / (tp[s] + fn[s] + 1e-8)
        f1 = 2 * pre * rec / (pre + rec + 1e-8) * 100
        f1s.append(f1)

    return {
        "Acc": acc,
        "Edit": edit,
        "F1@0.1": f1s[0],
        "F1@0.25": f1s[1],
        "F1@0.5": f1s[2]
    }
