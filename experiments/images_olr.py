
import torch
from torch import nn

from experiments.autils import Conv2dSameSize

from nde import distributions, transforms, flows
import utils
import nn as nn_

import matplotlib
# matplotlib.use('Agg')

from loguru import logger


from lightning import LightningModule
from experiments import autils
from experiments.autils import Conv2dSameSize

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import optim



class Preprocess:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float() # Already in [0,255]
        else:
            img = img * 255. # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits)) # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)

def create_transform_step(num_channels,
                          hidden_channels, actnorm, coupling_layer_type, spline_params,
                          use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob):
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = nn_.ConvResidualNet(in_channels=in_channels,
                                      out_channels=out_channels,
                                      hidden_channels=hidden_channels,
                                      num_blocks=num_res_blocks,
                                      use_batch_norm=resnet_batchnorm,
                                      dropout_probability=dropout_prob)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()
        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = utils.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)

def create_transform(c, h, w,
                     levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing,
                     multi_scale, transform_step_kwargs: dict):
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, **transform_step_kwargs) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, **transform_step_kwargs) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(
            input_shape=(c,h,w),
            output_shape=(c*h*w,)
        ))
        mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                                shift=-0.5)
    elif preprocessing == 'realnvp':
        preprocess_transform = transforms.CompositeTransform([
            # Map to [0,1]
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            # Map into unconstrained space as done in RealNVP
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - alpha)),
            transforms.Logit()
        ])

    elif preprocessing == 'realnvp_2alpha':
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit()
        ])
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct])


def create_flow(
    c, h, w,
    create_transform_kwargs: dict,
    transform_step_kwargs: dict,
    # flow_checkpoint=None, 
    _log=logger,
    ):
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(
        c, h, w, 
        **create_transform_kwargs,
        transform_step_kwargs=transform_step_kwargs
    )

    flow = flows.Flow(transform, distribution)

    _log.info('There are {} trainable parameters in this model.'.format(
        utils.get_num_parameters(flow)))

    # if flow_checkpoint is not None:
    #     flow.load_state_dict(torch.load(flow_checkpoint))
    #     _log.info('Flow state loaded from {}'.format(flow_checkpoint))

    return flow


DEFAULT_TRANSFORM_KWARGS = {
    "levels": 3,
    "multi_scale": True,
    "alpha": 0.05,
    "num_bits": 8,
    "steps_per_level": 10,
    "hidden_channels": 256,
    "preprocessing": "glow",
}
DEFAULT_TRANSFORM_STEP_KWARGS = {
    "actnorm": True,
    "coupling_layer_type": "rational_quadratic_spline",
    "spline_params": {
        "num_bins": 4,
        "tail_bound": 1.0,
        "min_bin_width": 1e-3,
        "min_bin_height": 1e-3,
        "min_derivative": 1e-3,
        "apply_unconditional_transform": False,
    },
    "use_resnet": False,
    "num_res_blocks": 5,
    "resnet_batchnorm": True,
    "dropout_prob": 0.0,
}

class Flow(LightningModule):
    def __init__(
        self,
        dataset_dims: tuple = (1, 64, 64),
        data_key: str = "image",
        create_transform_kwargs: dict = DEFAULT_TRANSFORM_KWARGS,
        transform_step_kwargs: dict = DEFAULT_TRANSFORM_STEP_KWARGS,
        num_bits: int = 8,
        learning_rate: float = 5e-4,
        eta_min: float = 0.0,
        cosine_annealing: bool = True,
        temperatures: list = [0.5, 0.75, 1.0],
        warmup_fraction: float = 0.0,
        sample_interval: int = 1,
    ):

        super().__init__()
        self.dataset_dims = dataset_dims
        self.data_key = data_key

        self.flow = create_flow(
            *self.dataset_dims,
            create_transform_kwargs=create_transform_kwargs,
            transform_step_kwargs=transform_step_kwargs,
            )
        self.num_bits = num_bits
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.cosine_annealing = cosine_annealing
        self.temperatures = temperatures
        self.warmup_fraction = warmup_fraction
        self.sample_interval = sample_interval
    
    def forward(self, x):
        return self.flow.log_prob(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.data_key]
        log_prob = self.flow.log_prob(x)
        loss = -autils.nats_to_bits_per_dim(log_prob.mean(), *self.dataset_dims)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.data_key]
        log_prob = self.flow.log_prob(x)
        val_loss = -autils.nats_to_bits_per_dim(log_prob.mean(), *self.dataset_dims)
        self.log("val/loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.cosine_annealing:
            if self.warmup_fraction == 0.0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.trainer.max_steps,
                    eta_min=self.eta_min
                )
            else:
                scheduler = optim.CosineAnnealingWarmUpLR(
                    optimizer=optimizer,
                    warm_up_epochs=int(self.warmup_fraction * self.trainer.max_steps),
                    total_epochs=self.trainer.max_steps,
                    eta_min=self.eta_min
                )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def on_validation_epoch_end(self):
        if self.current_epoch % self.sample_interval != 0:
            return

        with torch.no_grad():
            fig, axs = plt.subplots(1, len(self.temperatures), figsize=(4 * len(self.temperatures), 4))
            for temperature, ax in zip(self.temperatures, axs.flat):
                noise = self.flow._distribution.sample(64) * temperature
                samples, _ = self.flow._transform.inverse(noise)
                samples = Preprocess(self.num_bits).inverse(samples) # TODO implement Preprocess for OLR
                autils.imshow(make_grid(samples, nrow=8), ax)
                ax.set_title(f'T={temperature:.2f}')
            
            self.logger.experiment.add_figure("samples", fig, global_step=self.global_step)
            plt.close(fig)
