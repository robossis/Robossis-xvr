import os
from datetime import datetime
from pathlib import Path
from random import choice

import click
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import DoubleGeodesicSE3, MultiscaleNormalizedCrossCorrelation2d
from diffdrr.pose import convert
from diffdrr.registration import PoseRegressor
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

import wandb

from ..utils import XrayAugmentations, XrayTransforms, get_random_pose, render


@click.command(context_settings=dict(show_default=True, max_content_width=120))
@click.option(
    "-c",
    "--ckptpath",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint from which to resume training",
)
@click.option(
    "--project",
    default="diffpose",
    type=str,
    help="WandB project name",
)
def restart(ckptpath, project):
    """Restart training from a checkpoint."""
    ckpt = torch.load(ckptpath, weights_only=False)
    config = ckpt["config"]

    # Set up logging and train the model
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project=project, config=config)
    train_model(ckpt, config, run)


def train_model(ckpt, config, run):
    # Load all CT volumes
    volumes = []
    inpath = Path(config["inpath"])
    niftis = [inpath] if inpath.is_file() else sorted(inpath.glob("*.nii.gz"))
    for filepath in tqdm(niftis, desc="Reading CTs..."):
        subject = read(filepath, orientation=config["orientation"])
        volumes.append(subject.volume.data.squeeze().to(dtype=torch.float32))

    # Initialize deep learning modules
    model, drr, transforms, optimizer, scheduler = initialize(ckpt, config, subject)

    # Initialize the loss function
    imagesim = MultiscaleNormalizedCrossCorrelation2d([None, 9], [0.5, 0.5])
    geodesic = DoubleGeodesicSE3(config["sdd"])

    # Set up augmentations
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    augmentations = XrayAugmentations()

    # Train the model
    for epoch in range(ckpt["epoch"], config["n_epochs"] + 1):
        for _ in tqdm(range(config["n_batches_per_epoch"]), desc=f"Epoch {epoch}"):
            # Sample a random volume for this batch
            volume = choice(volumes).cuda()

            # Sample a batch of random poses
            pose = get_random_pose(config).cuda()

            # Render random DRRs and apply transforms
            contrast = contrast_distribution.sample().item()
            img, _, _ = render(drr, pose, volume, contrast)
            with torch.no_grad():
                img = augmentations(img)
            img = transforms(img)

            # Regress the poses and render the predicted DRRs
            pred_pose = model(img)
            pred_img, _, _ = render(drr, pred_pose, volume, contrast)

            # Compute the loss
            mncc = imagesim(img, pred_img)
            rgeo, tgeo, dgeo = geodesic(pose, pred_pose)
            loss = 1 - mncc + config["weight_geo"] * dgeo

            # Optimize the model
            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            # Log metrics
            wandb.log(
                {
                    "mncc": mncc.mean().item(),
                    "dgeo": dgeo.mean().item(),
                    "rgeo": rgeo.mean().item(),
                    "tgeo": tgeo.mean().item(),
                    "loss": loss.mean().item(),
                }
            )

        # Checkpoint the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "date": datetime.now(),
                    "config": config,
                },
                f"{config['outpath']}/{run.name}_{epoch:04d}.pth",
            )


class PoseFinetuner(torch.nn.Module):
    def __init__(self, config, model_state_dict):
        super().__init__()

        self.parameterization = config["parameterization"]
        self.convention = config["convention"]

        #
        model = PoseRegressor(
            model_name=config["model_name"],
            parameterization=config["parameterization"],
            convention=config["convention"],
            norm_layer=config["norm_layer"],
            height=config["height"],
        )
        model.load_state_dict(model_state_dict)
        self.backbone = model.backbone

        #
        self.xyz_regressor = torch.nn.Sequential(
            torch.nn.Linear(model.xyz_regression.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, model.xyz_regression.out_features),
        )

        self.rot_regressor = torch.nn.Sequential(
            torch.nn.Linear(model.rot_regression.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, model.rot_regression.out_features),
        )

    def forward(self, x):
        x = self.backbone(x)
        rot = self.rot_regressor(x)
        xyz = self.xyz_regressor(x)
        return convert(
            rot, xyz, parameterization=self.parameterization, convention=self.convention
        )


def initialize(ckpt, config, subject):
    # Initialize the pose regression model
    model = PoseFinetuner(
        config,
        ckpt["model_state_dict"],
        # model_name=config["model_name"],
        # pretrained=config["pretrained"],
        # parameterization=config["parameterization"],
        # convention=config["convention"],
        # norm_layer=config["norm_layer"],
        # height=config["height"],
    ).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    # Initialize a DRR renderer with a placeholder subject
    drr = DRR(
        subject,
        sdd=config["sdd"],
        height=config["height"],
        delx=config["delx"],
        reverse_x_axis=config["reverse_x_axis"],
        renderer=config["renderer"],
    ).cuda()
    transforms = XrayTransforms(config["height"])

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * config["n_batches_per_epoch"],
        config["n_epochs"] * config["n_batches_per_epoch"]
        - 5 * config["n_batches_per_epoch"],
    )  # Warmup for 5 epochs, then taper off

    # Load the checkpoint
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return model, drr, transforms, optimizer, scheduler
