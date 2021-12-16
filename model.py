import models
import torch
import networks

if __name__ == '__main__':
    model = {}
    model['UnetAdaptiveBins'] = models.UnetAdaptiveBins.build(100)
    model['pose_encoder'] = networks.ResnetEncoder(
                    18,# num_layers
                    True,# self.opt.weights_init == "pretrained",
                    num_input_images=2)
    model['pose'] = networks.PoseDecoder(
                    models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
    x = torch.rand(1, 3, 352, 704)
    y = torch.rand(1, 3, 352, 704)
    bins, pred = model['UnetAdaptiveBins'](x)
    
    print(bins.shape, pred.shape)      