## 非内置模型加载示例

服务使用NVIDIA Triton inference server构建，支持Tensorflow graphdef, Tensorflow savedmodel, Pytorch, TensorRT, ONNX等多种模型。非内置模型需要按Triton[模型仓库](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)的结构组织好，并且在加载时在`DFS目录`选择模型仓库下面模型名称对应的目录。

``` 
 <model-repository-path>/
    <model-name>/  #选择这个目录
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>  #选择这个目录
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```
## 支持模型框架
|推理格式               | 说明                                 |
|--------------------- |--------------------------------------|
|tensorflow_graphdef   |TensorFlow深度学习框架graphdef模型格式  |
|tensorflow_savedmodel |TensorFlow深度学习框架savedmodel模型格式|
|pytorch_libtorch      |Pytorch深度学习框架的模型格式           |
|tensorrt_plan         |NVIDIA硬件支持的Tensorrt模型格式       |
|onnxruntime_onnx      | Onnxruntime支持的onnx模型格式         |

## 数据类型

下表显示Triton推理支持的张量数据类型。第一列对应于模型配置文件中的数据类型。接下来的四列显示了支持的模型框架相对应的数据类型。第六列标记为“API”，显示了TRITONSERVER C API、TRITONBACKEND C API、HTTP/REST协议和GRPC协议对应的数据类型。最后一列显示Python numpy库相对应的数据类型。

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

### Tensorflow graphdef示例
#### 1、模型文件结构
```
model_name
  ├── 1
  │   └── model.graphdef
  └── config.pbtxt

``` 
#### 2、模型配置文件
```
  platform: "tensorflow_graphdef"
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

### Tensorflow savedmodel示例
#### 1、模型文件结构
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
#### 2、模型配置文件
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

### Pytorch示例
#### 1、模型文件结构
```
model_name
  ├── 1
  │   └── model.pt
  └── config.pbtxt

``` 
#### 2、模型配置文件
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

### TensorRT示例
#### 1、模型文件结构
```
modle_name
  ├── 1
  │   └── model.plan
  └── config.pbtxt
``` 
#### 2、模型配置文件
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

### ONNX示例
#### 1、模型文件结构
```
modle_name
  ├── 1
  │   └── model.onnx
  └── config.pbtxt
``` 
#### 2、模型配置文件
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
