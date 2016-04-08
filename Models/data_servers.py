#Import functions from fuel
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel_transformers_image import RandomHorizontalSwap, MaximumImageDimensions
from fuel.transformers import Flatten, Cast, ScaleAndShift
from argparse import ArgumentParser
from fuel.server import start_server

#PARAMETERS
image_size = (128,128)
batch_size = 64

#Function to get and process the data
def create_data(data, size, batch_size):
    if data == "train":
        stream = DogsVsCats(('train',), subset=slice(0, 22500))
        port = 5560
    elif data == "valid":
        stream = DogsVsCats(('train',), subset=slice(22500, 25000))
        port = 5561

    stream = DataStream(stream, iteration_scheme=ShuffledScheme(stream.num_examples, batch_size))
    stream = MinimumImageDimensions(stream, image_size, which_sources=('image_features',))
    stream = MaximumImageDimensions(stream, image_size, which_sources=('image_features',))
    stream = RandomHorizontalSwap(stream, which_sources=('image_features',))
    stream = Random2DRotation(stream, which_sources=('image_features',))
    stream = ScaleAndShift(stream, 1./255, 0, which_sources=('image_features',))
    stream = Cast(stream, dtype='float32', which_sources=('image_features',))
    start_server(stream, port=port)
    #stream_flat = Flatten(stream_scale, which_sources=('image_features',))


if __name__ == "__main__":
    parser = ArgumentParser("Run a fuel data stream server.")
    parser.add_argument("--type", type=str, default="train",
                          help="Type of the dataset (Train, Valid)")
    args = parser.parse_args()
    create_data(args.type, image_size, batch_size)
