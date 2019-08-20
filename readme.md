<!--
 * @Author: rayenwang
 * @Date: 2019-08-20 11:38:36
 * @Description: 
 -->

# 1. 数据准备

## 文件格式

> 数据格式与MOT17数据集保持一致
dataset_name
&emsp;&emsp;├── train
&emsp;&emsp;&emsp;&emsp;├── sequence_name
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── img1
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 000001.jpg
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── ....jpg
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── det
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── det.txt
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── gt
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── gt.txt
&emsp;&emsp;&emsp;&emsp;├── sequence_name...
&emsp;&emsp;├── test
&emsp;&emsp;&emsp;&emsp;├── sequence_name
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── img1
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 000001.jpg
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── ....jpg
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── det
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── det.txt
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── gt
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── gt.txt
&emsp;&emsp;&emsp;&emsp;├── sequence_name...

## training

> 训练标签保存在 gt.txt
格式: frame_index,track_id,x,y,w,h,x,y,z (图像id, 跟踪id, bbox, 类别label, 是否参与训练, 可见比例)
数据标定完成后, 得到数据格式
train
&emsp;&emsp;├── sequence_name
&emsp;&emsp;&emsp;&emsp;├── 000001.jpg
&emsp;&emsp;&emsp;&emsp;├── 000001.rec
&emsp;&emsp;&emsp;&emsp;├── ...
&emsp;&emsp;├── sequence_name ...
执行脚本`python tool/format_gt.py`, 数据集格式化

## prediction

> 检测结果保存在 det.txt
格式: frame_index,track_id,x,y,w,h,cof,x,y,z (图像id, 跟踪id, bbox, 类别label, 置信度, 是否参与训练, 可见比例)
`python video_to_image.py` 提取视频帧
执行目标目标检测, 结果保存至***.log文件, 保存格式: name label confidence x1 y1 x2 y2 (图像路径, 类别, 置信度, bbox)
将检测文件放到test文件夹下:
test
&emsp;&emsp;├── sequence_name
&emsp;&emsp;&emsp;&emsp;├── 000001.jpg
&emsp;&emsp;&emsp;&emsp;├── ....jpg
&emsp;&emsp;├── sequence_name...
&emsp;&emsp;├── format_detection.py
&emsp;&emsp;├── ****.log
执行脚本`python format_detection.py` 数据集格式化
remove format_detection.py and ***.log

# 2. 使用说明

## config

> 配置文件
Config: 通用配置
&emsp;&emsp;包括数据集路径, 是否使用cuda, 以及输入图像大小等
TrainConfig: 训练参数
&emsp;&emsp; 加载预训练模型, 中断继续, lr, focal loss参数和log, save参数
EvalConfig: 测试参数
&emsp;&emsp; 加载模型路径, 记忆池大小, confidence阈值和片段时长筛选

## train

> `python tools/get_max_object.py` 统计单帧图像最多有几个目标, 并设置config文件中的max_object参数
> `python train_mot17.py` 开始训练, 默认保存至checkpoints文件夹

## predict

> 修改config文件中的model_path参数, 加载训练好的模型
> `python eval_mot17.py --type train`or test 得到跟踪结果, 默认保存在result文件夹下
> `python tools/select_fragmengt.py --type train`or test 根据时长筛选片段
> `python tools/show_fragment.py --type train`or test 筛选后的跟踪结果可视化

# 3. 模块注释

## network

sst_loss.py 计算训练loss
sst.py forward过程, 包括 baseline, extractor, selector, final 四部分
tracker.py predict阶段跟踪过程

## pipline

augmentations.py 训练过程数据增强
mot_eval_dataset.py predict过程数据读入
mot_train_dataset.py train过程数据读入

## pretrained

训练好的模型

## tools

format_detection.py 检测结果格式化, 用于处理测试集
format_gt.py 数据标定结果格式化, 用于处理训练集
get_max_object.py 统计数据集中单帧图像最多有多少目标
performance.py 计算mota,motp等多目标跟踪评测指标
select_fragment.py 根据片段时长筛选跟踪结果
show_fragment.py 跟踪片段可视化
show_gt.py ground_truth可视化, 数据标定检查
show_track.py 跟踪结果可视化, 显示整个视频
