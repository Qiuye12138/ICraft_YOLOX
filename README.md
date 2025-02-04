# `Icraft`版`YoloX`

## 一、背景

截止到`V1.1`版本，`Icraft`暂时只原生支持`ReLU`和`LeakyReLU`两种激活函数，而`YoloX`使用了`SiLU`激活函数

可行的解决方法有以下几种：

1. 等待`IcraftV2.0`上线。该版本新增了对若干中激活函数的原生支持，包括`SiLU`
2. 使用`CustomOp`软算子。`Icraft`支持自定义软算子，但由于使用`CPU`计算，且激活函数出现频繁，导致数据传输时间过长

4. 修改模型，使用`LeakyReLU`替换`SiLU`作为激活函数，需要重新训练

本工程提供方案三的参考代码

此外，本工程还将每个检测层精简至卷积为止，其余的`Sigmoid`、`Concat`操作均移至后处理中由`CPU`执行



## 二、使用步骤
### 2.1、下载数据集

根据`datasets\README.md`要求摆好数据集

### 2.2、安装

```bash
pip install -v -e .
```

### 2.3、训练

下载预训练模型`yolox_s.pth`

使用以下代码即可开始训练

```bash
python -m yolox.tools.train -n yolox-s -d 1 -b 32 --fp16 -o -c yolox_s.pth
```

使用以下代码即可恢复最近一次训练

```bash
python -m yolox.tools.train -n yolox-s -d 1 -b 32 --fp16 -o --resume
```

> 问题1：OOM Runtime Error is raised due to the huge memory cost during label assignment. CPU mode is applied in this batch.
>
> GPU显存不够，调小batch-size

### 2.4、确认推理正确

使用以下代码即可推理

```bash
python tools\demo.py image -n yolox-s -c YOLOX_outputs\yolox_s\latest_ckpt.pth --path assets\dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```

### 2.5、测试精度

使用以下代码即可测试精度

```bash
python -m yolox.tools.eval -n  yolox-s -c YOLOX_outputs\yolox_s\latest_ckpt.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

本次重训练精度为：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
```

> 问题1：ModuleNotFoundError: No module named 'yolox.layers.fast_cocoeval
>
> 去Visual Studio安装目录搜索cl.exe，将x64的那个路径加入环境变量

### 2.6、保存模型

使用以下代码即可保存模型

```bash
python tools\export_torchscript.py -n yolox-s -c YOLOX_outputs\yolox_s\best_ckpt.pth
```



## 三、修改部分

`requirements.txt`：删除`onnx`系列，安装不上且没用到

`yolox/exp/yolox_base.py`：将`silu`替换为`lrelu`；你可以在这里修改模型的其他设置

`yolox/models/network_blocks.py`：将`Focus`类内的切片操作用卷积等效替换

`yolox/models/yolo_head.py`：将`obj_output`计算顺序提至最前；删除掉卷积后的操作

`yolox/core/trainer.py`：将`Focus`类内的等效卷积的权重搬移到指定位置

