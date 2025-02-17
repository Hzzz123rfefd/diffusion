# diffusion
diffusion model for image generation

## Installation
Install the packages required for development.
```bash
conda create -n diffusion python=3.10
conda activate diffusion
git clone https://github.com/Hzzz123rfefd/diffusion.git
cd diffusion
pip install -r requirements.txt
```

## Usage
### Dataset
In this example, we use the MNIST dataset to train the diffusion model and conditional diffusion model, we download the MNIST dataset and process it into the required data format for the project with following script:
```bash
python datasets/MNIST/process.py --data_dir datasets/MNIST/MNIST_data/ --output_dir datasets/MNIST/images/ --train_dir mnist_train/
```

No matter what dataset you use, please convert it to the required dataset format for this project, as follows:
```json
{"image_path": "datasets/MNIST/images/1\\12241.png", "label": 1}
{"image_path": "datasets/MNIST/images/5\\13749.png", "label": 5}
```

### Trainning
An examplary training script is provided in `train.py`.
You can adjust the model parameters in `config/ddpm.yml` and `config/cddpm.yml`
In this project, we have prepared two diffusion models, one with conditions and one without conditions, You can change the model at `model_config_path`
```bash
python train.py --model_config_path config/ddpm.yml
```

### Generate images
Once you have trained your model, you can use the following script to generate images:
```bash
python example/generate_images.py --model_config_path  config/ddpm.yml --save_image_path assets/output.png --label -1
```
```bash
python example/generate_images.py --model_config_path  config/cddpm.yml --save_image_path assets/output.png --label 8
```
![visualization02](assets/cddpm_output.png)
## Related links
 * refer: https://github.com/TeaPearce/Conditional_Diffusion_MNIST.git

