def chronCheck(sec, nsec, click, pattern):
    # print(f'chronCheck: patt: {pattern}')
    if pattern == '<':  # in the future
        if sec - click[0] < 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in future of {int(sec)}.{int(nsec)}')
            return True
        elif sec - click[0] == 0 and nsec - click[1] < 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in future of {int(sec)}.{int(nsec)}')
            return True
        else:
            return False

    elif pattern == '>':  # in the past
        if sec - click[0] > 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in past of {int(sec)}.{int(nsec)}')
            return True
        elif sec - click[0] == 0 and nsec - click[1] > 0:
            # print(f'chronCheck: {click[0]}.{click[1]} in past of {int(sec)}.{int(nsec)}')
            return True
        else:
            return False

    else:
        print('error: pattern is not \'>\' or \'<\'')
        return False


def clickInView(sec, nsec, old_sec, old_nsec, click_px):
    # print('clickInView')
    valid = []
    if old_sec is None:
        return valid
    for i, click in enumerate(click_px):
        if chronCheck(old_sec, old_nsec, click, '<') and chronCheck(sec, nsec, click, '>'):
            valid.append(i)
            print(f'valid found: {click_px[i]}')
    return valid
