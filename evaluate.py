import random
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from stretchbev.data import prepare_dataloaders
from stretchbev.metrics import IntersectionOverUnion, PanopticMetric
from stretchbev.trainer import TrainingModule
from stretchbev.utils.instance import predict_instance_segmentation_and_trajectories
from stretchbev.utils.network import preprocess_batch

# 30mx30m, 100mx100m
EVALUATION_RANGES = {
    '30x30': (70, 130),
    '100x100': (0, 200)
}


def eval(checkpoint_path, dataroot, version):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model
    model.eval()

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(
            device)
        iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']  # [:, 3:]
        intrinsics = batch['intrinsics']  # [:, 3:]
        extrinsics = batch['extrinsics']  # [:, 3:]
        future_egomotion = batch['future_egomotion']  # [:, 3:]

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            #  Evaluate with mean prediction
            output = model.multi_sample_inference(image, intrinsics, extrinsics, future_egomotion, num_samples=1,
                                                  future_distribution_inputs=future_distribution_inputs)[0]
        labels = {k: v[:, 3:] for k, v in labels.items()}

        #  Consistent instance seg
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False, make_consistent=True
        )

        segmentation_pred = output['segmentation'].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])
            panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                  labels['instance'][..., limits, limits].contiguous()
                                  )

            iou_metrics[key](segmentation_pred[..., limits, limits].contiguous(),
                             labels['segmentation'][..., limits, limits].contiguous()
                             )
            # iou_scores = iou_metrics[key].compute()
            # print(iou_scores[1].item())
            # iou_metrics[key].reset()

    results = {}
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')

    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    eval(args.checkpoint, args.dataroot, args.version)
