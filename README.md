
使用 TensorRT 进行推理
支持模型：Yolo v8

（1）借鉴 TensorRT_Pro 思路，使用混合内存封装 device 与 host memory

（2）适配新版 TensorRT API

（3）双缓冲机制，并行预处理与推理，减少阻塞