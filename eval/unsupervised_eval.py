import numpy as np
from scipy.optimize import linear_sum_assignment


def _filter_exclusions(pred_labels: np.ndarray, gt_labels: np.ndarray, excl_cls):
    if excl_cls is None:
        return pred_labels, gt_labels

    if isinstance(excl_cls, (list, tuple, set)):
        if len(excl_cls) == 0:
            return pred_labels, gt_labels
        if len(excl_cls) != 1:
            raise ValueError(f"ASOT-aligned excl_cls expects single class, got {excl_cls}")
        excl_cls = list(excl_cls)[0]

    mask = (gt_labels != excl_cls)
    return pred_labels[mask], gt_labels[mask]


def _pred_to_gt_match(pred_labels: np.ndarray, gt_labels: np.ndarray):
    pred_uniq = np.unique(pred_labels)
    gt_uniq = np.unique(gt_labels)

    affinity = np.zeros((len(pred_uniq), len(gt_uniq)), dtype=np.float64)
    for pi, p in enumerate(pred_uniq):
        p_mask = (pred_labels == p)
        for gi, g in enumerate(gt_uniq):
            affinity[pi, gi] = np.logical_and(p_mask, gt_labels == g).sum()

    cost = -affinity
    pi_opt, gi_opt = linear_sum_assignment(cost)

    pred_opt = pred_uniq[pi_opt]
    gt_opt = gt_uniq[gi_opt]
    return pred_opt, gt_opt


def _sorted_mapping_items(pred_to_gt: dict):
    return sorted(pred_to_gt.items(), key=lambda x: x[0])


def _remap_pred_with_mapping(pred_labels: np.ndarray, pred_to_gt: dict) -> np.ndarray:
    out = np.full(pred_labels.shape, fill_value=-999999, dtype=np.int64)
    for p, g in _sorted_mapping_items(pred_to_gt):
        out[pred_labels == p] = g
    return out


def eval_mof(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = _filter_exclusions(pred_labels, gt_labels, exclude_cls)

    if len(gt_labels_) == 0:
        return 0.0, ({} if pred_to_gt is None else pred_to_gt)

    if pred_to_gt is None:
        pred_opt, gt_opt = _pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt.tolist(), gt_opt.tolist()))
    else:
        items = _sorted_mapping_items(pred_to_gt)
        pred_opt = np.array([k for k, v in items])
        gt_opt = np.array([v for k, v in items])

    true_pos = 0
    for p, g in zip(pred_opt, gt_opt):
        true_pos += np.logical_and(pred_labels_ == p, gt_labels_ == g).sum()

    return float(true_pos) / float(len(gt_labels_)), pred_to_gt


def eval_miou(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = _filter_exclusions(pred_labels, gt_labels, exclude_cls)

    if len(gt_labels_) == 0:
        return 0.0, ({} if pred_to_gt is None else pred_to_gt)

    if pred_to_gt is None:
        pred_opt, gt_opt = _pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt.tolist(), gt_opt.tolist()))
    else:
        items = _sorted_mapping_items(pred_to_gt)
        pred_opt = np.array([k for k, v in items])
        gt_opt = np.array([v for k, v in items])

    class_tp = []
    class_union = []

    for p, g in zip(pred_opt, gt_opt):
        tp = np.logical_and(pred_labels_ == p, gt_labels_ == g).sum()
        un = np.logical_or(pred_labels_ == p, gt_labels_ == g).sum()
        class_tp.append(tp)
        class_union.append(un)

    denom = len(np.unique(gt_labels_))
    mean_iou = sum([tp / un for tp, un in zip(class_tp, class_union)]) / float(denom)
    return float(mean_iou), pred_to_gt


def _get_segments(frame_labels: np.ndarray):
    if len(frame_labels) == 0:
        return [], [], []

    labels = []
    starts = []
    ends = []

    last = frame_labels[0]
    start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != last:
            labels.append(int(last))
            starts.append(start)
            ends.append(i)
            start = i
            last = frame_labels[i]
    labels.append(int(last))
    starts.append(start)
    ends.append(len(frame_labels))
    return labels, starts, ends


def _levenshtein(p, y, norm=True):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros((m_row + 1, n_col + 1), dtype=np.float64)

    for i in range(m_row + 1):
        D[i, 0] = i
    for j in range(n_col + 1):
        D[0, j] = j

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(
                    D[i - 1, j] + 1,
                    D[i, j - 1] + 1,
                    D[i - 1, j - 1] + 1
                )

    if norm:
        if max(m_row, n_col) == 0:
            return 100.0
        return (1.0 - D[-1, -1] / max(m_row, n_col)) * 100.0
    return float(D[-1, -1])


def eval_edit(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = _filter_exclusions(pred_labels, gt_labels, exclude_cls)

    if len(gt_labels_) == 0:
        return 0.0, ({} if pred_to_gt is None else pred_to_gt)

    if pred_to_gt is None:
        _, pred_to_gt = eval_mof(pred_labels_, gt_labels_, n_videos=1, exclude_cls=None, pred_to_gt=None)

    pred_mapped = _remap_pred_with_mapping(pred_labels_, pred_to_gt)

    p_lab, _, _ = _get_segments(pred_mapped)
    g_lab, _, _ = _get_segments(gt_labels_)

    edit = _levenshtein(p_lab, g_lab, norm=True)
    return float(edit), pred_to_gt


def _segment_f1_overlap(pred_labels, gt_labels, overlap, pred_to_gt):
    pred_mapped = _remap_pred_with_mapping(pred_labels, pred_to_gt)

    p_label, p_start, p_end = _get_segments(pred_mapped)
    g_label, g_start, g_end = _get_segments(gt_labels)

    tp = 0.0
    fp = 0.0
    hits = np.zeros(len(g_label), dtype=np.int32)

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        intersection = np.maximum(intersection, 0)

        iou = (intersection / np.maximum(union, 1e-8)) * np.array(
            [p_label[j] == g_label[x] for x in range(len(g_label))],
            dtype=np.float64
        )

        idx = np.argmax(iou) if len(iou) > 0 else -1
        if idx >= 0 and iou[idx] >= overlap and not hits[idx]:
            tp += 1.0
            hits[idx] = 1
        else:
            fp += 1.0

    fn = float(len(g_label) - hits.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def eval_f1_overlap(pred_labels, gt_labels, n_videos, overlap=0.1, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = _filter_exclusions(pred_labels, gt_labels, exclude_cls)

    if len(gt_labels_) == 0:
        return 0.0, ({} if pred_to_gt is None else pred_to_gt)

    if pred_to_gt is None:
        _, pred_to_gt = eval_mof(pred_labels_, gt_labels_, n_videos=1, exclude_cls=None, pred_to_gt=None)

    f1 = _segment_f1_overlap(pred_labels_, gt_labels_, overlap=overlap, pred_to_gt=pred_to_gt)
    return float(f1), pred_to_gt


def eval_f1_sampling(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None,
                     n_sample=15, n_exper=50, eps=1e-8, rng: np.random.RandomState = None):
    pred_labels_, gt_labels_ = _filter_exclusions(pred_labels, gt_labels, exclude_cls)

    if len(gt_labels_) == 0:
        return 0.0, ({} if pred_to_gt is None else pred_to_gt)

    if rng is None:
        rng = np.random.RandomState(0)

    if pred_to_gt is None:
        pred_opt, gt_opt = _pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt.tolist(), gt_opt.tolist()))
    else:
        items = _sorted_mapping_items(pred_to_gt)
        pred_opt = np.array([k for k, v in items])
        gt_opt = np.array([v for k, v in items])

    n_actions = len(np.unique(gt_labels_))

    boundaries = np.where(gt_labels_[1:] - gt_labels_[:-1])[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(gt_labels_) - 1]))

    tp_agg = 0.0
    segments_count = 0

    for it in range(n_exper):
        for lo, up in zip(boundaries[:-1], boundaries[1:]):
            sample_idx = rng.randint(lo, up + 1, size=n_sample)
            gt_lab = gt_labels_[lo]

            if gt_lab in gt_opt:
                pred_lab = pred_opt[gt_opt == gt_lab]
                tp = (pred_labels_[sample_idx] == pred_lab).sum()
            else:
                tp = 0.0

            if (tp / float(n_sample)) > 0.5:
                tp_agg += 1.0

            if it == 0:
                segments_count += 1

    precision = tp_agg / (float(n_videos) * float(n_actions) * float(n_exper))
    recall = tp_agg / (float(segments_count) * float(n_exper) + eps)
    f1 = 2.0 * (precision * recall) / (precision + recall + eps)
    return float(f1), pred_to_gt


def compute_asot_metrics_per_full(preds_per_video, gts_per_video, exclude_cls=None,
                                  n_sample=15, n_exper=50, seed=0):
    assert len(preds_per_video) == len(gts_per_video), "pred/gt video count mismatch"
    B = len(preds_per_video)

    if B == 0:
        return {
            "MoF_per": 0.0, "MoF_full": 0.0,
            "mIoU_per": 0.0, "mIoU_full": 0.0,
            "F1_per": 0.0, "F1_full": 0.0,
            "Edit_per": 0.0, "Edit_full": 0.0,
            "F1@10_per": 0.0, "F1@10_full": 0.0,
            "F1@25_per": 0.0, "F1@25_full": 0.0,
            "F1@50_per": 0.0, "F1@50_full": 0.0,
        }

    mof_list, miou_list, f1_list = [], [], []
    edit_list, f10_list, f25_list, f50_list = [], [], [], []

    for i in range(B):
        p = np.asarray(preds_per_video[i], dtype=np.int64).reshape(-1)
        g = np.asarray(gts_per_video[i], dtype=np.int64).reshape(-1)

        rng = np.random.RandomState(seed + 10007 * i)

        mof_i, p2g_i = eval_mof(p, g, n_videos=1, exclude_cls=exclude_cls, pred_to_gt=None)
        miou_i, _ = eval_miou(p, g, n_videos=1, exclude_cls=exclude_cls, pred_to_gt=p2g_i)
        f1_i, _ = eval_f1_sampling(
            p, g, n_videos=1, exclude_cls=exclude_cls, pred_to_gt=p2g_i,
            n_sample=n_sample, n_exper=n_exper, rng=rng
        )
        edit_i, _ = eval_edit(p, g, n_videos=1, exclude_cls=exclude_cls, pred_to_gt=p2g_i)
        f10_i, _ = eval_f1_overlap(p, g, n_videos=1, overlap=0.10, exclude_cls=exclude_cls, pred_to_gt=p2g_i)
        f25_i, _ = eval_f1_overlap(p, g, n_videos=1, overlap=0.25, exclude_cls=exclude_cls, pred_to_gt=p2g_i)
        f50_i, _ = eval_f1_overlap(p, g, n_videos=1, overlap=0.50, exclude_cls=exclude_cls, pred_to_gt=p2g_i)

        mof_list.append(mof_i)
        miou_list.append(miou_i)
        f1_list.append(f1_i)
        edit_list.append(edit_i)
        f10_list.append(f10_i)
        f25_list.append(f25_i)
        f50_list.append(f50_i)

    mof_per = float(np.mean(mof_list))
    miou_per = float(np.mean(miou_list))
    f1_per = float(np.mean(f1_list))
    edit_per = float(np.mean(edit_list))
    f10_per = float(np.mean(f10_list))
    f25_per = float(np.mean(f25_list))
    f50_per = float(np.mean(f50_list))

    p_all = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1) for x in preds_per_video], axis=0)
    g_all = np.concatenate([np.asarray(x, dtype=np.int64).reshape(-1) for x in gts_per_video], axis=0)

    rng_full = np.random.RandomState(seed)

    mof_full, p2g_full = eval_mof(p_all, g_all, n_videos=B, exclude_cls=exclude_cls, pred_to_gt=None)
    miou_full, _ = eval_miou(p_all, g_all, n_videos=B, exclude_cls=exclude_cls, pred_to_gt=p2g_full)
    f1_full, _ = eval_f1_sampling(
        p_all, g_all, n_videos=B, exclude_cls=exclude_cls, pred_to_gt=p2g_full,
        n_sample=n_sample, n_exper=n_exper, rng=rng_full
    )
    edit_full, _ = eval_edit(p_all, g_all, n_videos=B, exclude_cls=exclude_cls, pred_to_gt=p2g_full)
    f10_full, _ = eval_f1_overlap(p_all, g_all, n_videos=B, overlap=0.10, exclude_cls=exclude_cls, pred_to_gt=p2g_full)
    f25_full, _ = eval_f1_overlap(p_all, g_all, n_videos=B, overlap=0.25, exclude_cls=exclude_cls, pred_to_gt=p2g_full)
    f50_full, _ = eval_f1_overlap(p_all, g_all, n_videos=B, overlap=0.50, exclude_cls=exclude_cls, pred_to_gt=p2g_full)

    return {
        "MoF_per": mof_per,
        "MoF_full": float(mof_full),
        "mIoU_per": miou_per,
        "mIoU_full": float(miou_full),
        "F1_per": f1_per,
        "F1_full": float(f1_full),
        "Edit_per": edit_per,
        "Edit_full": float(edit_full),
        "F1@10_per": f10_per,
        "F1@10_full": float(f10_full),
        "F1@25_per": f25_per,
        "F1@25_full": float(f25_full),
        "F1@50_per": f50_per,
        "F1@50_full": float(f50_full),
    }

