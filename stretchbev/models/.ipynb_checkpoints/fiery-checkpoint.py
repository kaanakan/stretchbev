import torch
import torch.nn as nn

from fiery.models.encoder import Encoder
from fiery.models.srvp_models import VGG64Encoder, VGG64Decoder
from fiery.models.res_models import SmallEncoder, SmallDecoder, ConvNet
from fiery.models.temporal_model import TemporalModelIdentity, TemporalModel
from fiery.models.distributions import DistributionModule
from fiery.models.future_prediction import FuturePrediction
from fiery.models.decoder import Decoder
from fiery.layers.temporal import SpatialGRU
from fiery.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from fiery.models import model_utils
from fiery.utils.geometry import cumulative_warp_features, calculate_birds_eye_view_parameters, VoxelsSumming


class Fiery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.min_log_sigma = self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA
        self.max_log_sigma = self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA
        self.skipco = self.cfg.MODEL.SMALL_ENCODER.SKIPCO

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM

        if self.cfg.MODEL.SUBSAMPLE:
            assert self.cfg.DATASET.NAME == 'lyft'
            self.receptive_field = 3
            self.n_future = 5

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)
        self.srvp_encoder = SmallEncoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE)
        self.srvp_decoder = SmallDecoder(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.SMALL_ENCODER.FILTER_SIZE, self.skipco)


        self.q_y = ConvNet(self.cfg.MODEL.ENCODER.OUT_CHANNELS*self.receptive_field, self.cfg.MODEL.ENCODER.OUT_CHANNELS*2, self.cfg.MODEL.FIRST_STATE.NUM_LAYERS) # # in_channels: y*conditioning_frames, out_channels: y*2 (mean, sigma), num_layers

        # residual update predictor
        self.dynamics = ConvNet(self.cfg.MODEL.ENCODER.OUT_CHANNELS+self.cfg.MODEL.DISTRIBUTION.LATENT_DIM, self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.DYNAMICS.NUM_LAYERS) # in_channels: y+z, out_channels: y, num_layers

        # inference of z, this is for processing posterior samples before sampling distribution parameters
        # q_z = MLP(LSTM(x)) in srvp, this is the LSTM 
        self.inf_z = FuturePrediction(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.ENCODER.OUT_CHANNELS+6, n_gru_blocks=2, n_res_layers=1)
        # SpatialGRU(self.cfg.MODEL.ENCODER.OUT_CHANNELS+6, self.cfg.MODEL.ENCODER.OUT_CHANNELS) 

        # posterior sampling
        self.q_z = ConvNet(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.DISTRIBUTION.LATENT_DIM*2, self.cfg.MODEL.DISTRIBUTION.POSTERIOR_LAYERS) # in_channels: y, out_channels: z*2, num_layers
        self.p_z = ConvNet(self.cfg.MODEL.ENCODER.OUT_CHANNELS, self.cfg.MODEL.DISTRIBUTION.LATENT_DIM*2, self.cfg.MODEL.DISTRIBUTION.PRIOR_LAYERS) # in_channels: y, out_channels: z*2, num_layers


        # Decoder
        self.decoder = Decoder(
            in_channels=self.cfg.MODEL.ENCODER.OUT_CHANNELS,
            n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS),
            predict_future_flow=self.cfg.INSTANCE_FLOW.ENABLED,
        )

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)
    

    def forward(self, image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs=None, noise=None, nt=None):
        output = {}

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()#.contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        x = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)
        #print('LSS output shape', x.shape)
        # LSS output shape torch.Size([1, 3, 64, 200, 200])

        srvp_x, skips = self.srvp_encode(x) # this will encode features into a more meaningful (?) space
        #print('srvp encoding done', srvp_x.shape)
        y_0, q_y0_params = self.infer_first_state(srvp_x[:, :self.receptive_field].contiguous()) # this will create the first state
        # print('first state is created', y_0.shape)

        # print('future', future_distribution_inputs.shape)
        ys, zs, q_z_params, p_z_params, residuals = self.srvp_generate(y_0, srvp_x,  future_distribution_inputs[:, :self.receptive_field].contiguous(), nt=nt) 
        # print('states are created', ys.shape)
        # above line will return intermediate ys, zs, distribution parameters and residual changes


        generated_srvp_x = self.srvp_decode(ys, skips) # this will decode intermediate states into LSS feature space
        # print('srvp decoding is done', generated_srvp_x.shape)

        bev_output = self.decoder(generated_srvp_x) # this will decode LSS features into FIERY outputs

        
        output = {
            'bev_output':bev_output,
            'generated_srvp_x': generated_srvp_x,
            'lss_outs':x,
            'q_z_params':q_z_params,
            'p_z_params':p_z_params,
            'q_y0_params':q_y0_params,
            # 'residuals':residuals
        }

        # output should contain:
        #   bev_output, generated FIERY outputs
        #   generated_srvp_x: predicted features
        #   x: LSS outputs
        #   q_z_params: posterior parameters
        #   p_z_params: prior parameters
        #   residuals: 
        return output

    def srvp_decode(self, x, skip):
        # content vector is missing. 
        """
        Decodes SRVP intermediate states into LSS output space
        x: SRVP intermediate states, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
        torch.Tensor: [batch, seq_len, channels, height, width]
        """
        # print('decode', x.shape)
        b, t, c, h, w = x.shape
        _x = x.view(b*t, c, h, w)
        if skip is not None:
            skip = [s.unsqueeze(1).expand(b, t, *s.shape[1:]) for s in skip]
            # print([s.shape for s in skip])
            skip = [s.reshape(t * b, *s.shape[2:]) for s in skip]
        # Decode and reshape
        # print('x',_x.shape)
        # print([s.shape for s in skip])
        x_out = self.srvp_decoder(_x, skip=skip)
        return x_out.view(b, t, *x_out.shape[1:])


    def srvp_encode(self, x):
        """
        Encodes LSS's outputs
        x: LSS encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
        Returns:
        torch.Tensor: [batch, seq_len, channels, height, width]
        """
        b, t, c, h, w = x.shape
        _x = x.view(b*t, c, h, w)
        hx, skips = self.srvp_encoder(_x, return_skip=True)
        hx = hx.view(b, t, *hx.shape[1:])
        if self.skipco:
            if self.training:
                # When training, take a random frame to compute the skip connections
                tx = torch.randint(t, size=(b,)).to(hx.device)
                index = torch.arange(b).to(hx.device)
                # print(tx, index)
                skips = [s.view(b, t, *s.shape[1:])[index, tx] for s in skips]
            else:
                # When testing, choose the last frame
                skips = [s.view(b, t, *s.shape[1:])[:, -1] for s in skips]
        else:
            skips = None
        return hx, skips


    def infer_first_state(self, x, deterministic=False):
        """
        Creates the first state from the conditioning observation
        x: encoded features, torch.Tensor, [batch, seq_len, channels, height, width]
        deterministic: If true, it will return only the means otherwise a sample from Normal distribution
        Returns:
        First state of the model-> y, torch.Tensor, [batch, channels, height, width]
        """
        # Q1: will the first state be stochastic?
        # Q2: are we going to sample a different noise for each position
        # print('y0 create', x.shape)
        b, t, c, h, w = x.shape
        _x = x.view(b, t*c, h, w)
        # print(_x.shape)
        q_y0_params = self.q_y(_x)
        y_0 = model_utils.rsample_normal(q_y0_params, max_log_sigma=self.max_log_sigma, min_log_sigma=self.min_log_sigma)
        return y_0, q_y0_params

    def infer_z(self, hx):
        """
        Infers z from the SpatialGRU's output
        hx: output of SpatialGRU, torch.Tensor, [batch, channels, height, width]
        Returns:
        z: torch.Tensor, [batch, channels, height, width]
        qz_params: torch.Tensor, [batch, channels*2, height, width]
        """
        q_z_params = self.q_z(hx)
        z = model_utils.rsample_normal(q_z_params, max_log_sigma=self.max_log_sigma, min_log_sigma=self.min_log_sigma)
        return z, q_z_params


    def residual_step(self, y, z, dt=1):
        # QUESTION: should we keep euler steps thing?
        # Are we going to create states for not integer timesteps
        """
        y: intermediate state, torch.Tensor, [batch, channels, height, width]
        z: latent noise, torch.Tensor, [batch, channels, height, width]
        Returns:
        updated y: torch.Tensor, [batch, channels, height, width]
        """
        # print('y', y.shape, 'z', z.shape)
        res_step = self.dynamics(torch.cat([y, z], 1))
        y_forward = y + dt*res_step
        return y_forward, res_step
    

    def srvp_generate(self, y_0, x, future_inputs, nt=None):
        """
        Generates intermediate states with residual updates
        y_0: created first state, [batch, seq_len, channels, height, width]
        x: encoded features, [batch, seq_len, channels, height, width]
        Returns:
        ys: intermediate states, [batch, seq_len, channels, height, width] # dim might change
        zs: used gaussian noise, [batch, seq_len - 1, channels, height, width] # dim might change
        q_z_params: posterior distribution parameters, [batch, seq_len - 1, 2*channels, height, width] # dim might change
        p_z_params: prior distribution parameters, [batch, seq_len - 1, 2*channels, height, width] # dim might change
        residuals: residual changes, [batch, seq_len - 1, channels, height, width] # dim might change
        """
        total_time = self.receptive_field + self.n_future if nt is None else nt
        #print('total, receptive, future', total_time, self.receptive_field, self.n_future)
        ys = [y_0]
        z, q_z_params, p_z_params = [], [], []
        res = []
        # print('x', x.shape, 'future_inp', future_inputs.shape)
        hx_z = self.inf_z(torch.cat([x, future_inputs], dim=2)) # encodes srvp_encoder's output temporally
        # print('gru out', hx_z.shape)
        y_tm1 = y_0
        for t in range(1, total_time):

            # prior distribution
            p_z_t_params = self.p_z(y_tm1)
            p_z_params.append(p_z_t_params)

            if t < self.receptive_field: #self.training or t < self.receptive_field:
                # print('smth wrong')
                #print(t, 'posterior')
                # observations are avaliable
                z_t, q_z_t_params = self.infer_z(hx_z[:, t])
                q_z_params.append(q_z_t_params)
            else:
                #print(t, 'prior')
                assert not self.training
                z_t = model_utils.rsample_normal(p_z_t_params, max_log_sigma=self.max_log_sigma, min_log_sigma=self.min_log_sigma)
            # Residual step
            # print(z_t)
            y_t, res_t = self.residual_step(y_tm1, z_t)
            # Update previous latent state
            y_tm1 = y_t
            ys.append(y_t)
            res.append(res_t)

        y = torch.stack(ys, 1)
        z = torch.stack(z, 1) if len(z) > 0 else None
        q_z_params = torch.stack(q_z_params, 1) if len(q_z_params) > 0 else None
        p_z_params = torch.stack(p_z_params, 1) if len(p_z_params) > 0 else None
        res = torch.stack(res, 1)
        return y, z, q_z_params, p_z_params, res



    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, b, s)
        return x

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([present_features, future_features], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution
