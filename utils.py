import random


def get_padded_image(image, row_pad, col_pad, fill):
    dim = len(image) + 2 * col_pad
    padded_image = [
        *[[fill for _ in range(dim)]] * row_pad,
        *[[fill] * col_pad + list(r) + [fill] * col_pad for r in image],
        *[[fill for _ in range(dim)]] * row_pad
    ]

    return padded_image


def zerofill(row, col):
    return [
        [0 for _ in range(row)]
        for _ in range(col)
    ]


def clamp(value, f, c):
    return max(min(value, c), f)


def relu(v):
    return max(v, 0)


def gen_mask(dimension):
    return [
        [
            random.normalvariate(0, 1) for _ in range(dimension)
        ] for _ in range(dimension)
    ]


def apply_convolution(image, mask):
    pad_size = int(len(mask) / 2)
    pad_dim = len(image) + pad_size * 2

    padded_image = zerofill(pad_dim, pad_dim)

    for pr in range(len(image)):
        for pc in range(len(image[0])):
            for mr in range(len(mask)):
                for mc in range(len(mask[0])):
                    padded_image[
                        pr + mr][pc + mc] += mask[mr][mc] * image[pr][pc]

    trimmed = padded_image[pad_size:-pad_size]
    trimmed = [
        [relu(c) for c in row[pad_size:-pad_size]]
        for row in trimmed
    ]

    return trimmed


def apply_pool(image, pool_size=2):
    pooled_dim = int(len(image) / pool_size)
    pooled_image = zerofill(pooled_dim, pooled_dim)

    for pr in range(pooled_dim):
        for pc in range(pooled_dim):
            # max pooling
            x, y = (pr * pool_size, pc * pool_size)
            pooled_image[pr][pc] = max(
                image[x][y:y + pool_size - 1]
                + image[x + pool_size - 1][y:y + pool_size - 1]
            )

    return pooled_image
