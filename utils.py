

def bounding_rect_to_corners(rect):
    corners = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
    return corners


def corners_to_bounding_box(rect):
    box = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]
    return box

