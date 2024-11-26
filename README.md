# Through-Wall Imaging based on WiFi Channel State Information

### Proposed Architecture

<img src="resources/pipeline.svg" alt="Pipeline Diagram" width="1000" height="300">
Based on the Mixture-of-Products-of-Experts Variational Autoencoder (MoPoE-VAE) from https://github.com/thomassutter/MoPoE

### Through-Wall Imaging Example
Ground truth image (left) and image reconstructed from WiFi CSI captured in a through-wall scenario (right).

https://github.com/user-attachments/assets/3572af8e-8958-48f3-b844-8c3c61b54e65

### Paper
**Strohmayer J., Sterzinger R., Stippel C. and Kampel M.**, “Through-Wall Imaging Based On WiFi Channel State Information,” 2024 IEEE International Conference on Image Processing (ICIP), Abu Dhabi, United Arab Emirates, 2024, pp. 4000-4006, doi: https://doi.org/10.1109/ICIP51287.2024.10647775.

BibTeX:
```
@INPROCEEDINGS{Strohmayer10647775,
  author={Strohmayer, Julian and Sterzinger, Rafael and Stippel, Christian and Kampel, Martin},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Through-Wall Imaging Based On WiFi Channel State Information}, 
  year={2024},
  volume={},
  number={},
  pages={4000-4006},
  doi={10.1109/ICIP51287.2024.10647775}}
```

### Prerequisites
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset
Get the wificam dataset from https://zenodo.org/uploads/11554280 and put it in the `/data` directory.

### Training & Testing

**Training** | Command for training a *MoPoE-VAE* model with the aggegration method *concatenation* and *temporal encoding* enabled for 50 epochs:

```
python3 train.py --name mopoevae_ct --data data/wificam/ --epochs 50 --am concat --tenc --device 0
```
Logging is disabled by default. You can enable wandb logging by updating your credentials in *train.py* and passing the `--log` argument.

**Testing** | Command for testing the trained model:

```
python3 train.py --name mopoevae_ct --data data/wificam/ --epochs 50 --am concat --tenc --device 0 --test
```
During Testing, ground truth and generated images will be logged in the *runs/mopoevae_ct/out/* directory which can be turned into a video via ffmpeg:
```
ffmpeg -framerate 100 -i %d.png -c:v libx264 -pix_fmt yuv420p demo.mp4
```
