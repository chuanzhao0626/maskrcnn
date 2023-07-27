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
### 3. Input and output of the model
The input and output configurations of the model configuration file must match the properties defined by the model structure. The model definition property contains the input and output names, data types, and dimension sizes. The configuration needs to be defined in the following format:
```
input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [-1]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [-1]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [-1]
   }
 ```
### 4. Datatypes
The following table shows the tensor datatypes supported by
Triton. The first column shows the name of the datatype as it appears
in the model configuration file. The next four columns show the
corresponding datatype for supported model frameworks. If a model
framework does not have an entry for a given datatype, then Triton
does not support that datatype for that model. The sixth column,
labeled "API", shows the corresponding datatype for the TRITONSERVER C
API, TRITONBACKEND C API, HTTP/REST protocol and GRPC protocol. The
last column shows the corresponding datatype for the Python numpy
library.

|Model Config  |TensorRT      |TensorFlow    |ONNX Runtime  |PyTorch  |API      |NumPy         |
|--------------|--------------|--------------|--------------|---------|---------|--------------|
|TYPE_BOOL     | kBOOL        |DT_BOOL       |BOOL          |kBool    |BOOL     |bool          |
|TYPE_UINT8    | kUINT8       |DT_UINT8      |UINT8         |kByte    |UINT8    |uint8         |
|TYPE_UINT16   |              |DT_UINT16     |UINT16        |         |UINT16   |uint16        |
|TYPE_UINT32   |              |DT_UINT32     |UINT32        |         |UINT32   |uint32        |
|TYPE_UINT64   |              |DT_UINT64     |UINT64        |         |UINT64   |uint64        |
|TYPE_INT8     | kINT8        |DT_INT8       |INT8          |kChar    |INT8     |int8          |
|TYPE_INT16    |              |DT_INT16      |INT16         |kShort   |INT16    |int16         |
|TYPE_INT32    | kINT32       |DT_INT32      |INT32         |kInt     |INT32    |int32         |
|TYPE_INT64    |              |DT_INT64      |INT64         |kLong    |INT64    |int64         |
|TYPE_FP16     | kHALF        |DT_HALF       |FLOAT16       |         |FP16     |float16       |
|TYPE_FP32     | kFLOAT       |DT_FLOAT      |FLOAT         |kFloat   |FP32     |float32       |
|TYPE_FP64     |              |DT_DOUBLE     |DOUBLE        |kDouble  |FP64     |float64       |
|TYPE_STRING   |              |DT_STRING     |STRING        |         |BYTES    |dtype(object) |
|TYPE_BF16     |              |              |              |         |BF16     |              |
## Model configuration file example display
### Example description

##### 1. The name of the model file "model_name" can be modified. "1" in the "model_name" directory must be a number.
##### 2. The model file name in the version directory (such as directory 1 in the example) must be the same as that in the example. For example, the TensorRT model file must be named Model.plan.
##### 3. Multiple versions of this model can be stored in the model directory, but the inference service loads only the model weights of the latest version. For example, if there are three versions, 1,3,7, the inference service loads only the model weights of version 7.
##### 4. The config.pbtxt file is the model configuration file and its name cannot be changed. For configuration information, see the preceding.
##### 5. The number of parameters in "input" and "output" in the model configuration file is not fixed, and must correspond to the number of inputs and outputs in the model structure.
##### 6. The dims parameter in the model configuration file indicates the dimensions of the input and output of the model.

### Tensorflow graphdef example
#### 1. Model file structure
```
 model_name
   ├── 1
   │   └── model.graphdef
   └── config.pbtxt
``` 
#### 2. Model configuration file
```
 platform: "tensorflow_graphdef"
 max_batch_size: 8
 input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [224,224,3]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [32,32]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
```

### Tensorflow savedmodel example
#### 1. Model file structure
```
 model_name
    ├── 1
    │   └── model.savedmodel
    │          └── saved_model.pb
    │          ├── variables
    │                └── variables.data-00000-of-00001
    │                ├── variables.index
    ├── config.pbtxt
``` 
#### 2. Model configuration file
```
 platform: "tensorflow_savedmodel"
 max_batch_size: 8
 input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
```

### Pytorch example
#### 1. Model file structure
```
model_name
   ├── 1
   │   └── model.pt
   └── config.pbtxt
``` 
#### 2. Model configuration file
```
 platform: "pytorch_libtorch"
 max_batch_size: 8
 input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
```

### TensorRT example
#### 1. Model file structure
```
 modle_name
   ├── 1
   │   └── model.plan
   └── config.pbtxt
``` 
#### 2. Model configuration file
```
 platform: "tensorrt_plan"
 max_batch_size: 8
 input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
```

### ONNX example
#### 1. Model file structure
```
 modle_name
   ├── 1
   │   └── model.onnx
   └── config.pbtxt
``` 
#### 2. Model configuration file
```
 platform: "onnxruntime_onnx"
 max_batch_size: 8
 input [
   {
     name: "input0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   },
   {
     name: "input1"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
 output [
   {
     name: "output0"
     data_type: TYPE_FP32
     dims: [ 16 ]
   }
 ]
```
