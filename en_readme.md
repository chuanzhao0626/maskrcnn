## Custom model loading examples

A service uses a NVIDIA Triton inference server. It supports Tensorflow graphdef, Tensorflow savedmodel, Pytorch, TensorRT, ONNX, etc. Loaded models shall be organized into Triton's [model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html). Select the model name directory under the model repository for `DFS folder`.

``` 
 <model-repository-path>/
    <model-name>/  #select this folder
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>  #select this folder/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
``` 
## Model file configuration instructions
Triton inference requires a model configuration file, Details refer to [parameter configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html). The following is an introduction to the characteristics of each field.

### 1. support model type description
The Triton Inference service supports model file inference for a variety of deep learning frameworks, which corresponds to the "platform" field in the model configuration file. The fields are described as follows:

| Model type           | Description                                                    |
|----------------------|----------------------------------------------------------------|
|tensorflow_graphdef   | The TensorFlow deep learning framework graphdef model format   |
|tensorflow_savedmodel | The TensorFlow deep learning framework savedmodel model format |
|pytorch_libtorch      | The Pytorch deep learning framework model format               |
|tensorrt_plan         | The Tensorrt model format supported by NVIDIA hardware         |
|onnxruntime_onnx      | The onnx model supported by Onnxruntime                        |
### 2. max_batch_size configuration description
The *max_batch_size* property indicates the maximum batch size that
the model supports for the types of
batching that can be exploited
by Triton. If the model's batch dimension is the first dimension, and
all inputs and outputs to the model have this batch dimension,In this case *max_batch_size* should be set to a value
greater-or-equal-to 1 that indicates the maximum batch size that
Triton should use with the model.

For models that do not support batching, or do not support batching in
the specific ways described above, *max_batch_size* must be set to
zero.
##### Supports batch model file configuration
```
  # Supports the maximum (64,32,32,3) batch model input inference, the maximum output dimension is (64,10), and the batch range is an integer [1-64]
  platform: "tensorrt_plan"
  max_batch_size: 64
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [32,32,3]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [10]
    }
  ]
```
##### Unsupport batch model file configuration
```
  #(32,32,3) Model inference input, output dimension is (10)
  platform: "tensorrt_plan"
  max_batch_size: 0
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [32,32,3]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [10]
    }
  ]
```
### Tensorflow graphdef example

### Tensorflow savedmodel example

### Pytorch example

### TensorRT example

### ONNX example
