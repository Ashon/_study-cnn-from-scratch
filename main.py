from mnist import MNISTImages
from mnist import MNISTLabels

from utils import apply_convolution
from utils import apply_pool
from utils import gen_mask


image_train = MNISTImages('files/train-images-idx3-ubyte')
label_train = MNISTLabels('files/train-labels-idx1-ubyte')

print(image_train.magic_number)
print(image_train.n_items)
print(image_train.n_rows, image_train.n_cols)

pixels_train = image_train.get_image_by_idx(0)
image_train.print_pixels(pixels_train)
print(label_train.get_label_by_idx(0))
print()

image_test = MNISTImages('files/t10k-images-idx3-ubyte')
label_test = MNISTLabels('files/t10k-labels-idx1-ubyte')

# print(image_test.magic_number)
# print(image_test.n_items)
# print(image_test.n_rows, image_test.n_cols)

# pixels_test = image_test.get_image_by_idx(0)
# image_test.print_pixels(pixels_test)
# print(label_test.get_label_by_idx(0))
# print()

# print(pixels_train)

# from utils import get_padded_image
# mask_dim = len(mask)
# mask_row_padding = int(mask_dim / 2)
# mask_col_padding = int(mask_dim / 2)

# pad_fill = 255
# pad_dimension = len(pixels_train) + 2
# padded_pixels = get_padded_image(
#     pixels_train, mask_row_padding, mask_col_padding, pad_fill)
# print(padded_pixels)
# image_train.print_pixels(padded_pixels)

n_conv = 1

batch_size = 20

mask_size = 3
pool_size = 2


network = [
    (apply_convolution, n_conv),
    (apply_convolution, n_conv),
]

# Test
# image = image_train.get_image_by_idx(0)
# conv = apply_convolution(image, gen_mask(3))
# image_train.print_pixels(conv)

maps = [[]] + [[] for _ in network]

print(maps)
for i in range(int(image_train.n_items / batch_size)):
    for b in range(batch_size):
        samples = [
            image_train.get_image_by_idx(i * batch_size + b)
            for _ in range(n_conv)
        ]
        maps[0] += samples

        for idx, layer in enumerate(network):
            convolutions = [
                layer[0](maps[idx][j], gen_mask(mask_size))
                for j in range(layer[1])
            ]

            maps[idx + 1] += convolutions

    print(
        f'Train {i / int(image_train.n_items / batch_size) * 100:.2f}% // '
        f'{i * batch_size}/{image_train.n_items}'
    )
