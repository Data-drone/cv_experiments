try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import logging

pipeline_logger = logging.getLogger(__name__ + '.data_pipeline')

# Basic DALI accelerated pipeline
# Does a random rotate and crop
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_shard=0, total_shards=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_shard, num_shards=total_shards, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
            
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.8, 1.25],
                                                      random_area=[0.1, 1.0],
                                                      num_attempts=100)

        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            # from https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                            mean=[0.50707516 * 255,0.48654887 * 255,0.44091784 * 255], 
                                            std=[0.26733429 * 255,0.25643846 * 255,0.27615047 * 255]) 

        self.coin = ops.CoinFlip(probability=0.5)

        # add a rotate
        self.rotate = ops.Rotate(device='gpu', interp_type=types.INTERP_NN) 
        self.rotate_range = ops.Uniform(range = (-7, 7)) # 7 degrees either way
        self.rotate_coin = ops.CoinFlip(probability=0.075) # 7.5% chance

        pipeline_logger.info('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        # for mirror
        rng = self.coin()
        # for rotate
        angle_range = self.rotate_range()
        prob_rotate = self.rotate_coin()
        
        self.jpegs, self.labels = self.input(name="Reader") # load in files
        images = self.decode(self.jpegs) # decode
        images = self.rotate(images, angle=angle_range, mask=prob_rotate) # rotate
        images = self.res(images) # resize
        output = self.cmnp(images.gpu(), mirror=rng) # crop / mirror / normalise (crop is random 50%)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_shard=0, total_shards=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_shard, num_shards=total_shards, random_shuffle=False)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            # from https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
                                            mean=[0.50707516 * 255,0.48654887 * 255,0.44091784 * 255], 
                                            std=[0.26733429 * 255,0.25643846 * 255,0.27615047 * 255]) 

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]