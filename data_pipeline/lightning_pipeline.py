try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# Data Transform class for dali
# this mirrors the whole Albumentations transform chain that we have
class DaliTransformsTrainPipeline(Pipeline):
    def __init__(self, batch_size, device, data_dir, mean, std, device_id=0, 
                    shard_id=0, num_shards=1, num_threads=4, seed=0):
        super(DaliTransformsTrainPipeline, self).__init__(batch_size, num_threads, device_id, seed)

        # should we make this drive the device flags?
        self.reader = ops.FileReader(file_root=data_dir, shard_id=shard_id, 
                                        num_shards=num_shards, random_shuffle=True)

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB, memory_stats=True)
        self.resize = ops.Resize(device=device, size=[200,300], interp_type=types.INTERP_TRIANGULAR)
        self.centrecrop = ops.Crop(device=device, crop=[100,100])
        self.randomcrop = ops.RandomResizedCrop(device=device, size=[80,80])

        self.hz_coin = ops.CoinFlip(probability=0.5)
        self.horizontalflip = ops.Flip(device=device)

        self.rotate_angle = ops.Uniform(range=[-90,90])
        self.rotate_coin = ops.CoinFlip(probability=0.5)
        self.rotate = ops.Rotate(device=device, keep_size=True)
        
        self.vt_coin = ops.CoinFlip(probability=0.5)
        self.verticalflip = ops.Flip(device=device, horizontal=0)

        self.normalize = ops.CropMirrorNormalize(device=device, dtype=types.FLOAT, 
                                                output_layout=types.NCHW) #,
                                                #mean=mean*255, std=std*255)

        #self.normalize = ops.Normalize(device=device, dtype=types.FLOAT)#, mean=mean, stddev=std)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)
        #self.transpose_torch = ops.Transpose(device=device, perm=[0,1,2,3], output_layout=types.NCHW)


    def define_graph(self):

        # randomness variables
        hz_rng = self.hz_coin()
        rt_coin = self.rotate_coin()
        rt_angle = self.rotate_angle()
        vr_rng = self.vt_coin()

        jpegs, labels = self.reader(name='Reader')
        
        labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)
        
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        images = self.centrecrop(images)
        images = self.randomcrop(images)
        images = self.horizontalflip(images, horizontal=hz_rng)
        # We multiple the uniform dist sample with the 0/1 from the coin flip
        #images = self.rotate(images, angle=rt_coin*rt_angle)
        images = self.verticalflip(images, vertical=vr_rng)
        images = self.normalize(images)
        
        return (images, labels)



class DaliTransformsValPipeline(Pipeline):
    def __init__(self, batch_size, device, data_dir, mean, std, device_id=0, 
                    shard_id=0, num_shards=1, num_threads=4, seed=0):
        super(DaliTransformsValPipeline, self).__init__(
            batch_size, num_threads, device_id, seed)

        # should we make this drive the device flags?
        self.reader = ops.FileReader(file_root=data_dir, shard_id=shard_id, 
                                        num_shards=num_shards, random_shuffle=False)

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB, memory_stats=True)
        self.resize = ops.Resize(device=device, size=[200,300], interp_type=types.INTERP_TRIANGULAR)

        self.normalize = ops.CropMirrorNormalize(device=device, dtype=types.FLOAT, 
                                                output_layout=types.NCHW) #,
                                                #mean=mean*255, std=std*255)

        #self.normalize = ops.Normalize(device=device, dtype=types.FLOAT)#, mean=mean, stddev=std)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)
        

    def define_graph(self):

        jpegs, labels = self.reader(name="Reader")
        
        labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)
        
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        images = self.normalize(images)
        
        return (images, labels)