import struct


class MNISTBase(object):
    _metadata_offset = 0

    def __init__(self, image_file):
        self._image_file = image_file
        self._file = open(self._image_file, 'rb')

        self._meta = struct.unpack(
            f'>{int(self._metadata_offset / 4)}I',
            self._file.read(self._metadata_offset))

    @property
    def magic_number(self):
        return self._meta[0]

    @property
    def n_items(self):
        return self._meta[1]

    @property
    def n_rows(self):
        return self._meta[2]

    @property
    def n_cols(self):
        return self._meta[3]


class MNISTImages(MNISTBase):
    _metadata_offset = 16

    @property
    def n_pixels(self):
        return self.n_rows * self.n_cols

    def get_image_by_idx(self, idx: int):
        if not (-1 < idx < self.n_items):
            raise Exception('ImageIdxNotInRangeError')

        self._file.seek(
            self._metadata_offset + self.n_pixels * idx)

        pixels = struct.unpack(
            f'>{self.n_pixels}B',
            self._file.read(self.n_pixels))

        reshaped_pixels = [
            pixels[i:i + self.n_cols]
            for i in range(0, self.n_pixels, self.n_cols)
        ]

        return reshaped_pixels

    @staticmethod
    def print_pixels(pixels):
        for row in pixels:
            print(
                ''.join([
                    f"\033[48;5;{int(i / 256 * 24) + 232}m \033[0m"
                    for i in row
                ])
            )


class MNISTLabels(MNISTBase):
    _metadata_offset = 8

    def get_label_by_idx(self, idx: int):
        if not (-1 < idx < self.n_items):
            raise Exception('ImageIdxNotInRangeError')

        self._file.seek(self._metadata_offset + idx)

        label = struct.unpack(f'>B', self._file.read(1))[0]
        return label
