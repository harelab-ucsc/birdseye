
def read_label_file(load_name):
    try:
        f = open(f"{load_name}", "rb")
        # print(f'    Label loaded: frame index {frame_index}, {load_name}')
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        while True:
            # print(i)
            mask = f.readline()
            tmp = mask.split()
            if len(tmp) == 0:
                break
            c1 += float(tmp[-1])
            c2 += (1.0 - float(tmp[-1]))
            c3 += 1.0
        f.close()
    except FileNotFoundError:
        pass
    return c1, c2, c3


if __name__ == '__main__':
    load_name = '/home/harey/farm_1/labels.txt'
    print(load_name)
    p, n, t = read_label_file(load_name)
    print(f' Positive Frames: {p} \n Negative Frames: {n} \n Total Frames: {t}')