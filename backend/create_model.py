import torch
import torchvision.models as models
import torch.nn as nn

# Create a simple model based on ResNet
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 30)  # 30 classes for our animals

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(model,                     # model being run
                 dummy_input,                # model input
                 "model/model.onnx",         # where to save the model
                 export_params=True,         # store the trained parameter weights inside the model file
                 opset_version=11,           # the ONNX version to export the model to
                 do_constant_folding=True,   # whether to execute constant folding for optimization
                 input_names=['input'],      # the model's input names
                 output_names=['output'],    # the model's output names
                 dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                             'output': {0: 'batch_size'}}) 