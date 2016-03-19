#Inspired by fuel.transformers.RandomFixedSizeCrop
#http://www.scipy-lectures.org/advanced/image_processing/index.html#basic-image

from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
from skimage.transform import resize

import numpy as np

class ScikitResize(SourcewiseTransformer, ExpectsAxisLabels):
    def __init__(self, data_stream, image_shape, crop=True, seed=1,**kwargs):
        """

        crop : if crop is true, this object will crop image to the biggest square shape before resizing

        """
        self.image_shape = image_shape
        self.crop = crop
        if crop is True:
            self.rng = np.random.RandomState(seed)

        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(ScikitResize, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
#        print("a batch transform is starting")
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        if isinstance(source, np.ndarray) and source.ndim == 4:
            return [self.transform_source_example(im, source_name)
                    for im in source]

        elif all([isinstance(b, np.ndarray) and b.ndim == 3 for b in source]):
            return [self.transform_source_example(im, source_name)
                    for im in source]
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)

        if not isinstance(example, np.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")


        if self.crop is True :

 #           print("ex :::: ",example.shape)
            image_height, image_width = example.shape[1:]
            crop_size = min((image_height, image_width))

            windowed_height, windowed_width = (crop_size,)*2

            if image_height < windowed_height or image_width < windowed_width:
                raise ValueError("can't obtain ({}, {}) window from image "
                                 "dimensions ({}, {})".format(
                                     windowed_height, windowed_width,
                                     image_height, image_width))
            if image_height - crop_size > 0:
                off_h = self.rng.random_integers(0, image_height - crop_size)
            else:
                off_h = 0
            if image_width - crop_size > 0:
                off_w = self.rng.random_integers(0, image_width - crop_size)
            else:
                off_w = 0
            ex=example[:, off_h:off_h + windowed_height,
                       off_w:off_w + windowed_width]
        else :
            ex = example

        num_channel = example.shape[0]
        resized = np.zeros((num_channel,self.image_shape[0],self.image_shape[1]))

        for x in range(num_channel):
            resized[x] = resize(ex[x], self.image_shape)
        return resized
