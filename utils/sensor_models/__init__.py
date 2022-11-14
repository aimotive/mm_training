import json

from .CameraBase import CameraBase
from .CameraPinhole import CameraPinhole
from .CameraEquirect import CameraEquirect
from .CameraPinholeDistorted import CameraPinholeDistorted
from .CameraMei import CameraMei

# register string version of model names
registered_models = [CameraPinhole, CameraPinholeDistorted, CameraMei, CameraEquirect]
registered_names = [CameraPinhole.model_name, CameraPinholeDistorted.model_name, CameraMei.model_name, CameraEquirect.model_name]
name_model_register = {name:model for name,model in zip(registered_names, registered_models)}

def make_from_json(fp):
    # WIP code
    json_dict = {}
    with open(fp, 'r') as stream:
        json_dict = json.load(stream)
    assert 'model_name' in json_dict
    if not json_dict['model_name'] in name_model_register:
        print(f"The json contains a model name not presen in the model register: {json_dict['model_name']}.")
        raise ValueError

    model = name_model_register[json_dict['model_name']]  # type: CameraBase
    return model.load_from_dict(json_dict)
