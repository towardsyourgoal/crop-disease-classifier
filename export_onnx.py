"""Export the best checkpoint (mobilenet_v3_small) to a single-file ONNX model for deployment."""
import onnx
import torch
import torch.nn as nn
import torchvision

NUM_CLASSES = 12

m = torchvision.models.mobilenet_v3_small(weights=None)
m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
m.load_state_dict(torch.load('mobilenet_v3_small.pt', map_location='cpu'))
m.eval()

torch.onnx.export(m, torch.randn(1, 3, 224, 224), 'model/tmp.onnx',
                  input_names=['image'], output_names=['logits'],
                  dynamic_axes={'image': {0: 'batch'}}, opset_version=18)

# merge any external .data file into one self-contained file
onnx.save_model(onnx.load('model/tmp.onnx'), 'model/mobilenet_v3_small.onnx', save_as_external_data=False)
