"""
from https://nms.readthedocs.io/en/latest/_modules/nms/felzenszwalb.html with some changes
Felzenszwalb et al implementation of NMS
"""

import numpy as np


def rect_areas(rects):
    rects = np.array(rects)
    w = rects[:, 2]
    h = rects[:, 3]
    return w * h


def rect_compare(rect1, rect2, area):
    xx1 = max(rect1[0], rect2[0])
    yy1 = max(rect1[1], rect2[1])
    xx2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    yy2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    return float(w * h) / area


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:, 0].argsort()[::-1]]
    else:
        npscores = npscores[npscores[:, 0].argsort()]

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()


def nms(boxes, scores, classes, **kwargs):
    top_k = kwargs.get('top_k', 0)
    assert 0 <= top_k

    score_threshold = kwargs.get('score_threshold', 0.3)
    assert 0 < score_threshold

    nms_threshold = kwargs.get('nms_threshold', 0.4)
    assert 0 < nms_threshold < 1

    if len(boxes) == 0:
        return []

    if scores is not None:
        assert len(scores) == len(boxes)

    pick = set()

    area = rect_areas(boxes)
    scores = get_max_score_index(scores, score_threshold, top_k, False)
    indices = np.array(scores, np.int32)[:, 1]

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        pick.add(i)
        suppress = [last]

        for pos in range(0, last):
            j = indices[pos]

            overlap = rect_compare(boxes[i], boxes[j], area[j])

            if overlap > nms_threshold and classes[i] == classes[j]:
                suppress.append(pos)

        indices = np.delete(indices, suppress)
    return pick


def non_max_suppression(boxes, nms_threshold):
    if len(boxes) == 0:
        return []

    if isinstance(boxes, list):
        boxes = np.array(boxes)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    rects = boxes[:, 0:4]
    scores = boxes[:, 4]
    classes = boxes[:, 5]

    output = []
    for i in nms(rects, scores, classes, nms_threshold=nms_threshold):
        output.append(list(boxes[i]))
    return output
