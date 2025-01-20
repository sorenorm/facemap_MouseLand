"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""

import os
import pickle
import time
from io import StringIO

import h5py
import numpy as np
import torch
from tqdm import tqdm

from facemap import utils

from . import datasets, facemap_network, model_loader, model_training
from . import pose_helper_functions as pose_utils
from . import transforms


class Pose:
    """
    Pose estimation for single video processing.

    Parameters
    ----------
    filenames: 2D-list
        List of filenames to be processed.
    bbox: list
        Bounding box for cropping the video [x1, x2, y1, y2]. If not set, the entire frame is used.
    bbox_set: bool
        Flag to indicate whether the bounding box has been set. Default is False.
    resize: bool
        Flag to indicate whether the video needs to be resized.
    add_padding: bool
        Flag to indicate whether the video needs to be padded. Default is False.
    gui: object
        GUI object.
    GUIobject: object
        GUI mainwindow object.
    net: object
        PyTorch model object.
    model_name: str
        Name of the model to be used for pose estimation. Default is None which uses the pre-trained model.
    """

    def __init__(
        self,
        filenames=None,
        bbox=[],
        bbox_set=False,
        resize=False,
        add_padding=False,
        gui=None,
        GUIobject=None,
        net=None,
        model_name=None,
    ):
        self.gui = gui
        self.GUIobject = GUIobject
        if self.gui is not None:
            self.filenames = self.gui.filenames
            self.batch_size = self.gui.batch_size_spinbox.value()
        else:
            self.filenames = filenames
            self.batch_size = 1
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(
            self.filenames
        )
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        if gui is not None:
            self.device = self.gui.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bbox = bbox
        self.bbox_set = bbox_set
        self.resize = resize
        self.add_padding = add_padding
        self.net = net
        self.model_name = model_name
        self.bodyparts = [
            "eye(back)",
            "eye(bottom)",
            "eye(front)",
            "eye(top)",
            "lowerlip",
            "mouth",
            "nose(bottom)",
            "nose(r)",
            "nose(tip)",
            "nose(top)",
            "nosebridge",
            "paw",
            "whisker(I)",
            "whisker(III)",
            "whisker(II)",
        ]

    def pose_prediction_setup(self):
        self.load_model()
        if not self.bbox_set:
            for i in range(len(self.Ly)):
                x1, x2, y1, y2 = 0, self.Ly[i], 0, self.Lx[i]
                self.bbox.append([x1, x2, y1, y2])
                if x2 - x1 != y2 - y1:
                    self.add_padding = True
                if x2 - x1 != 256 or y2 - y1 != 256:
                    self.resize = True
            self.bbox_set = True

    def set_model(self, model_selected=None):
        if model_selected is None:
            model_selected = (
                "Base model"
                if self.gui is None
                else self.gui.pose_model_combobox.currentText()
            )
        model_paths = model_loader.get_model_states_paths()
        if not model_paths:
            self.model_name = model_loader.get_basemodel_state_path()
        else:
            model_names = [
                os.path.splitext(os.path.basename(m))[0] for m in model_paths
            ]
            for model in model_names:
                if (model == model_selected) or (
                    model_selected == "Base model" and "facemap_model_state" in model
                ):
                    self.model_name = model_paths[model_names.index(model)]
                    break
        self.net.load_state_dict(torch.load(self.model_name, map_location=self.device))
        self.net.to(self.device)

    def load_model(self):
        model_params_file = model_loader.get_model_params_path()
        model_params = torch.load(model_params_file, map_location=self.device)
        channels = model_params["params"]["channels"]
        nout = len(self.bodyparts)
        self.net = facemap_network.FMnet(
            img_ch=1,
            output_ch=nout,
            labels_id=self.bodyparts,
            channels=channels,
            kernel=3,
            device=self.device,
        )
        self.set_model()

    def predict_landmarks(self, video_id, frame_ind=None):
        nchannels = 1
        if frame_ind is None:
            total_frames = self.cumframes[-1]
            frame_ind = np.arange(total_frames)
        else:
            total_frames = len(frame_ind)

        pred_data = torch.zeros(total_frames, len(self.bodyparts), 3)
        heatmaps_list = []
        self.net.eval()
        start = 0
        end = self.batch_size
        y1, _, x1, _ = self.bbox[video_id]
        inference_time = 0

        with tqdm(total=total_frames, unit="frame", unit_scale=True) as pbar:
            while start != total_frames:
                imall = np.zeros(
                    (self.batch_size, nchannels, self.Ly[video_id], self.Lx[video_id])
                )
                cframes = np.array(frame_ind[start:end])
                imall = utils.get_batch_frames(
                    cframes,
                    total_frames,
                    self.cumframes,
                    self.containers,
                    video_idx=video_id,
                    grayscale=True,
                )
                t0 = time.time()
                imall, postpad_shape, pads = transforms.preprocess_img(
                    imall,
                    self.bbox[video_id],
                    self.add_padding,
                    self.resize,
                    device=self.net.device,
                )
                xlabels, ylabels, likelihood, hm_pred = pose_utils.predict(
                    self.net, imall, smooth=False
                )
                heatmaps_list.append(hm_pred.cpu())
                xlabels, ylabels = transforms.adjust_keypoints(
                    xlabels,
                    ylabels,
                    crop_xy=(x1, y1),
                    padding=pads,
                    current_size=(256, 256),
                    desired_size=postpad_shape,
                )
                pred_data[start:end, :, 0] = xlabels
                pred_data[start:end, :, 1] = ylabels
                pred_data[start:end, :, 2] = likelihood
                inference_time += time.time() - t0
                pbar.update(self.batch_size)
                start = end
                end += self.batch_size
                end = min(end, total_frames)

        self.heatmaps = torch.cat(heatmaps_list, dim=0)
        metadata = {
            "batch_size": self.batch_size,
            "image_size": (self.Ly, self.Lx),
            "bbox": self.bbox[video_id],
            "total_frames": total_frames,
            "bodyparts": self.bodyparts,
            "inference_speed": total_frames / inference_time,
        }
        return pred_data, metadata

    def save_model(self, model_filepath):
        torch.save(self.net.state_dict(), model_filepath)
        model_loader.copy_to_models_dir(model_filepath)
        return model_filepath

    def save_data_to_hdf5(self, data, video_id, selected_frame_ind=None):
        scorer = "Facemap"
        bodyparts = self.bodyparts
        data_dict = {scorer: {}}
        indices = (
            np.arange(self.cumframes[-1])
            if selected_frame_ind is None
            else selected_frame_ind
        )
        for index, bodypart in enumerate(bodyparts):
            data_dict[scorer][bodypart] = {
                "x": data[:, index, 0][indices],
                "y": data[:, index, 1][indices],
                "likelihood": data[:, index, 2][indices],
            }

        basename, filename = os.path.split(self.filenames[0][video_id])
        videoname, _ = os.path.splitext(filename)
        hdf5_filepath = os.path.join(basename, videoname + "_FacemapPose.h5")
        with h5py.File(hdf5_filepath, "w") as f:
            self.save_dict_to_hdf5(f, "", data_dict)
        return hdf5_filepath

    def save_dict_to_hdf5(self, h5file, path, data_dict):
        for key, item in data_dict.items():
            if isinstance(item, dict):
                self.save_dict_to_hdf5(h5file, path + key + "/", item)
            else:
                h5file[path + key] = item

    def run(self):
        self.pose_prediction_setup()
        for video_id in range(len(self.filenames[0])):
            pred_data, metadata = self.predict_landmarks(video_id)
            savepath = self.save_data_to_hdf5(pred_data.cpu().numpy(), video_id)
            print(f"Saved keypoints: {savepath}")
