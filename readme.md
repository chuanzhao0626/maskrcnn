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
## 模型文件配置说明
Triton推理需要模型配置文件，详情可参考[参数配置](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)。以下是对各个字段的介绍。

### 1、支持模型类型说明
Triton推理服务支持多种深度学习框架的模型文件推理，其对应于模型配置文件中的“platform”字段。字段说明介绍如下：
|模型类型               | 说明                                 |
|--------------------- |--------------------------------------|
|tensorflow_graphdef   |TensorFlow深度学习框架graphdef模型格式  |
|tensorflow_savedmodel |TensorFlow深度学习框架savedmodel模型格式|
|pytorch_libtorch      |Pytorch深度学习框架的模型格式           |
|tensorrt_plan         |NVIDIA硬件支持的Tensorrt模型格式       |
|onnxruntime_onnx      | Onnxruntime支持的onnx模型格式         |
### 2、max_batch_size配置说明
max_batch_size属性表示模型推理支持的最大批处理大小。模型需满足批处理维度是第一个维度，并且模型的所有输入和输出都具有该批处理维度，同时满足上述条件便可在配置文件中添加批处理属性。这种情况下max_batch_size为大于或等于1的值。对于不支持批处理，或者不支持上述特定方式的批处理的模型，该属性必须设置为零。
##### 支持批处理模型文件配置
```
  #支持最大(64,32,32,3)的批处理模型输入推理，输出维度最大为(64,10)，批处理范围为[1-64]的整数
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
##### 不支持批处理模型文件配置
```
  #(32,32,3)模型推理输入，输出维度为(10)
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
### 3、模型的输入和输出
模型配置文件的输入和输出配置必须与模型结构定义的属性相匹配。模型定义属性包含输入和输出的名称、数据类型和维度大小，配置需定义如下格式：
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
### 4、数据类型
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
## 模型配置文件示例展示
### 示例说明
##### 1、“model_name”模型文件名称可以修改，“model_name”目录下的“1”为此模型的版本号，必须为数字。
##### 2、版本目录（如示例中“1”目录）下模型文件名称需与示例中的名称保持一致，例如TensorRT模型文件需命名为“model.plan”。
##### 3、模型目录里可存储多个此模型版本，但推理服务只加载最新版本号的模型权重，比如存在1,3,7三个版本，推理服务只加载版本7的模型权重。
##### 4、“config.pbtxt”文件为模型配置文件，文件名称不能修改。配置信息请参考上面内容。
##### 5、模型配置文件中的“input”和“output”中的参数个数是不固定的，需对应模型结构中的输入和输出数量。
##### 6、模型配置文件中的“dims”参数为模型输入和输出的维度大小。

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
