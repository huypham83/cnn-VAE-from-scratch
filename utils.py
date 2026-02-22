import cupy as cp

def get_im2col_idx(x_shape, kernel_height, kernel_width, padding=1, stride=1):
    N, C, H, W = x_shape

    out_height = (H + 2 * padding - kernel_height) // stride + 1
    out_width = (W + 2 * padding - kernel_width) // stride + 1

    i0 = cp.tile(cp.repeat(cp.arange(kernel_height), kernel_width), C)
    j0 = cp.tile(cp.arange(kernel_width), kernel_height * C)

    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = cp.repeat(cp.arange(C), kernel_height * kernel_width).reshape(-1, 1)
    return k, i, j

def im2col(x, kernel_height, kernel_width, padding=1, stride=1):
    N, C, H, W = x.shape
    x_padded = cp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    k, i, j = get_im2col_idx(x.shape, kernel_height, kernel_width, padding, stride)

    cols = x_padded[:, k, i, j]

    cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * C, -1)

    return cols

def col2im(cols, x_shape, kernel_height, kernel_width, padding=1, stride=1):
    N, C, H, W = x_shape
    H_pad, W_pad = H + padding * 2, W + padding * 2
    x_padded = cp.zeros((N, C, H_pad, W_pad))

    k, i, j = get_im2col_idx(x_shape, kernel_height, kernel_width, padding, stride)

    cols = cols.reshape(kernel_height * kernel_width * C, -1, N).transpose(2, 0, 1)

    cp.add.at(x_padded, (slice(None), k, i, j), cols)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
