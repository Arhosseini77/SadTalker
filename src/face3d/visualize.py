import scipy.io as scio
import torch
from tqdm import tqdm
from src.face3d.models.facerecon_model import FaceReconModel
import numpy as np


def gen_landmarks_list(args, device, first_frame_coeff, coeff_path, exp_dim=64):
    # Load coefficients
    coeff_first = scio.loadmat(first_frame_coeff)['full_3dmm']
    coeff_pred = scio.loadmat(coeff_path)['coeff_3dmm']

    # Prepare coefficients
    coeff_full = np.repeat(coeff_first, coeff_pred.shape[0], axis=0)
    coeff_full[:, 80:144] = coeff_pred[:, 0:64] # Expression coefficients
    coeff_full[:, 224:227] = coeff_pred[:, 64:67] # Translation coefficients
    coeff_full[:, 254:] = coeff_pred[:, 67:] # Other coefficients

    # Initialize face model
    facemodel = FaceReconModel(args)

    # Generate landmarks list
    landmarks_list = []
    last_computed_landmarks = None
    for k in tqdm(range(0, coeff_pred.shape[0], 2), 'Generating landmarks:'):
        cur_coeff_full = torch.tensor(coeff_full[k:k + 1], device=device)

        # Only compute landmarks for every third frame
        facemodel.forward(cur_coeff_full, device)
        predicted_landmark = facemodel.pred_lm.cpu().numpy().squeeze()

        # Use the computed landmarks for the next frame as well
        for _ in range(2):
            if k + _ < coeff_pred.shape[0]:
                landmarks_list.append(predicted_landmark)

    return landmarks_list


