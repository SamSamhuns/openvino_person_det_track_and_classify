# OpenVINO

Set up a conda/venv virtual env and install dependencies from  `requirements.txt`

## OpenVINO install and inference

### OpenVINO PyPi Package

Install with pip using: `pip install openvino` from <https://pypi.org/project/openvino/>

### OpenVINO Python Inference Samples

<https://github.com/odundar/openvino_python_samples>

## Running sample python scripts

### 1. Person/Object Detection

```bash
# image mode
$ python openvino_person_detection.py -m image -i PATH_TO_IMG -o OUTPUT_DIR
# video mode
$ python openvino_person_detection.py -m video -i PATH_TO_VID -o OUTPUT_DIR
# webcam mode
$ python openvino_person_detection.py -m webcam -o OUTPUT_DIR
```

### 2. Person Re-identification

```bash
# video mode
$ python openvino_person_reidentification.py -m video -i PATH_TO_VID -o OUTPUT_DIR
```

### 3. Person Action Counting (Punch/Kick/Normal)

```bash
# video mode
$ python openvino_person_action_counting.py -m video -i PATH_TO_VID -o OUTPUT_DIR
```

## Pretrained OpenVINO models

1.  **OpenVINO Model Documentation and Download**

<https://docs.openvinotoolkit.org/2021.3/omz_models_group_intel.html>

2.  **OpenVINO GitHub Public Model Repo**

<https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public>

3.  **Download Intel OpenVINO IR (Intermediate Representation) models**

-   OpenVino Workbench with Docker (Recommended) <https://docs.openvino.ai/latest/workbench_docs_Workbench_DG_Introduction.html>

-   Python `openvino-dev` pip pkg <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/README.md>

## Creating OpenVINO models from other NN frameworks

1.  **Convert to OpenVINO format with Docker**

Local repository available in `./google_teachable_machine_to_openvino`

<https://github.com/ojjsaw/teachable-machine-openvino.git>

2.  **OpenVINO training extensions**

<https://github.com/openvinotoolkit/training_extensions>

3.  **Deep Learning Workbench from DockerHub**

Download & Convert to IR models for OpenVINO Inference

<https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install_from_Docker_Hub.html>
