import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime
import matplotlib.pyplot as plt
import os,gc

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import RigidTransform
from diffdrr.pose import convert
from diffdrr.visualization import plot_drr

def mark_and_backproject(img, drr, pose):
    """
    Mark 2D points on a DRR and backproject them into 3D world coordinates.
    """
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Click landmark points on DRR. Close when done.")
    points = plt.ginput(n=-1, timeout=0)
    plt.close()

    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(drr.device)
    height = img.shape[-2]
    width = img.shape[-1]
    points_tensor[..., 1] = height - points_tensor[..., 1]
    points_tensor[..., 0] = width - points_tensor[..., 0]   

    fiducials_3d = drr.inverse_projection(pose, points_tensor)
    torch.save(fiducials_3d.cpu(), "xray_fiducials.pt")

    return fiducials_3d



def save_drr_as_dicom_white_background(
    img_tensor,           # (1, 1, H, W) or (1, H, W) torch tensor
    output_path: str,     # Output DICOM path
    sdd: float,           # Source to detector distance
    delx: float,          # Pixel spacing (mm)
    detector_origin: list, # [x0_mm, y0_mm]
    rows: int,            # Image height (H)
    cols: int             # Image width (W)
):
    # Convert tensor to normalized uint16 with white background
    img_np = img_tensor.squeeze().cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    img_inverted = 1.0 - img_norm  # dark bone, white background
    img_16bit = (img_inverted * 65535).astype(np.uint16)

    # Create file meta
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.1"
    meta.ImplementationVersionName = "PYDICOM 2.4.4"

    # Create DICOM dataset
    ds = FileDataset(output_path, {}, file_meta=meta, preamble=b"\0" * 128)
    dt = datetime.datetime.now()
    ds.StudyDate = dt.strftime("%Y%m%d")
    ds.StudyTime = dt.strftime("%H%M%S")

    ds.Modality = "OT"
    ds.PatientName = "Synthetic"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = meta.MediaStorageSOPClassUID

    # Image dimensions and type
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.PixelData = img_16bit.tobytes()

    # Display window
    ds.WindowCenter = 32896
    ds.WindowWidth = 65535

    # Geometry
    ds.DistanceSourceToDetector = "%.10f" % sdd
    ds.DetectorActiveOrigin = ["%.10f" % detector_origin[0], "%.10f" % detector_origin[1]]
    ds.PixelSpacing = ["%.10f" % delx, "%.10f" % delx]

    # Save DICOM
    ds.save_as(output_path)
    print(f"✅ DICOM saved to {output_path}")

ct_path = "SE000002.nii.gz"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# DRR configuration
sdd = 1000
height = 1200
width = 1200
delx = 0.25813333

# Load CT
subject = read(ct_path)
drr = DRR(subject, sdd=sdd, height=height, width=width, delx=delx).to(device)

gc.collect()
torch.cuda.empty_cache()

# Define pose
rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
xyz = torch.tensor([[0.0, 1000.0, 0.0]], device=device)


# Generate DRR
img = drr(rot, xyz, parameterization="euler_angles", convention="ZXY",)


# Save pose
pose = convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
print("True Pose:",pose.matrix)
# Save 4x4 pose matrix
torch.save(pose, "xray_pose.pt")


img_display = img.squeeze().cpu().numpy()

# 2. Call the function to interactively mark and backproject
fiducials_3d = mark_and_backproject(img_display, drr, pose)
print("Marked 3D Points (World Coordinates):\n", fiducials_3d)


dicom_output_path = "xray_image.dcm"
origin_x = -drr.detector.x0 * drr.detector.delx
origin_y = -drr.detector.y0 * drr.detector.dely
detector_origin = [origin_x, origin_y]

save_drr_as_dicom_white_background(
    img_tensor=img,               # (1, 1, H, W)
    output_path=dicom_output_path,
    sdd=sdd,
    delx=delx,                   # as per your metadata
    detector_origin=detector_origin,
    rows=height,                     # or img.shape[-2]
    cols=width                      # or img.shape[-1]
)   