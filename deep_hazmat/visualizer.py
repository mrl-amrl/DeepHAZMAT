import cv2


def draw_lines(image, points):
    for i in range(1, len(points)):
        s = points[i - 1]
        e = points[i]
        cv2.line(image, s, e, (255, 255, 255), 2)


def put_text(image, text, x, y, color, scale):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    text_height = int(text_size[0][1])
    cv2.putText(image, text, (x, y + text_height), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_box(image, x, y, w, h, color, name=None, thickness=2, padding=0):
    px = int(padding * w)
    py = int(padding * h)
    x -= px
    y -= py
    h += py * 2
    w += px * 2

    if name is not None:
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_width = int(text_size[0][0] * 1.05)
        cv2.rectangle(image, (x, y - 17), (x + text_width + 10, y), color, -1)
        cv2.putText(image, name, (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    offset = int(w / 20)
    x += offset
    y += offset
    w -= offset * 2
    h -= offset * 2

    mw = int(w / 4)
    mh = int(h / 4)
    cv2.line(image, (x, y), (x + mw, y), color, thickness)
    cv2.line(image, (x + w - mw, y), (x + w, y), color, thickness)

    cv2.line(image, (x, y), (x, y + mh), color, thickness)
    cv2.line(image, (x, y + h - mh), (x, y + h), color, thickness)

    cv2.line(image, (x, y + h), (x + mw, y + h), color, thickness)
    cv2.line(image, (x + w - mw, y + h), (x + w, y + h), color, thickness)

    cv2.line(image, (x + w, y), (x + w, y + mh), color, thickness)
    cv2.line(image, (x + w, y + h - mh), (x + w, y + h), color, thickness)
