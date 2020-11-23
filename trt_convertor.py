# -*- coding: utf-8 -*-
"""
Created on 2020.06.11

@author: LWS
"""
import tensorrt as trt

def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    # 声明一个TRT LOGGER，可以指定TRT日志的等级
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 如果是 ONNX 模型，这里的 EXPLICIT_BATCH 是必不可少的，否则 parse 时会报错提醒
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # 初始化一个 Builder， 创建一个空的 Network, 初始化 OnnxParser
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 * 1 << 30    # 指定最大工作空间 1 * 1 << 30 = 1GB
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        # parse ONNX file
        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            # parser.parse(model.read())
        print('Completed parsing of ONNX file')

        # build TRT engine
        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


def loadEngine2TensorRT(filepath):
    """
    保存好的 TensorRT engine可以通过以下代码进行反序列化，读取到内存中使用, 构建TensorRT inference engine
    """
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
