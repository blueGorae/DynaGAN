import os
from gdown import download as drive_download

google_drive_paths = {
    'shape_predictor_68_face_landmarks.dat': 'https://drive.google.com/uc?id=17kwWXLN9fA6acrBWqfuQCBdcc1ULmBc9',
    'e4e_ffhq_encode.pt': 'https://drive.google.com/uc?id=1O8ipkyMYHwCRmuaZBaO-KYZ9FYuH8Xnc',
    'ffhq.pt': 'https://drive.google.com/uc?id=1XQabKtkpMltyZkFYidX4jd8Zrii5eTyI&export=download',
    'ffhq_PCA.npz': 'https://drive.google.com/uc?id=13b81CBny0VgxWJWWEylNJkNbXuQ512ug&export=download',
    'afhqcat.pt': 'https://drive.google.com/uc?id=17K_U0IKaVKoQT4lJ6zf1h6ijfmrHSB7B&export=download',
    'afhqcat_PCA.npz': 'https://drive.google.com/uc?id=1_JiWz-8eiki-LFFF0Aerf8GpM6mpjpYR&export=download'
}


def download_weight(weight_path):
    if not os.path.isfile(weight_path) and (
            os.path.basename(weight_path) in google_drive_paths
    ):

        gdrive_url = google_drive_paths[os.path.basename(weight_path)]
        try:
            # drive_download(gdrive_url, weight_path, fuzzy=True)
            drive_download(gdrive_url, weight_path, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

