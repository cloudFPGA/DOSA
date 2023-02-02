import time
import numpy as np
import torch
from .tensorrt_quant import build_engine
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def export_onnx(model, onnx_file_path, input_shape):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(input_shape, device=device)
    torch.onnx.export(
        model, dummy_input, onnx_file_path, verbose=False, opset_version=13
    )


def find_engine_target_batch_size(engines_contexts, target_batch_size):
    for engine, context, input_shape in engines_contexts:
        if input_shape[0] == target_batch_size:
            return engine, context, input_shape
    return None


def run(context, inputs, outputs, bindings, stream, run_interval=None):
    for host, device in inputs:
        cuda.memcpy_htod_async(device, host, stream)

    run_interval = 0 if run_interval is None else run_interval
    runtime = -1
    start_time = time.time()
    while runtime < run_interval:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        runtime = time.time() - start_time

    for host, device in outputs:
        cuda.memcpy_dtoh_async(host, device, stream)

    stream.synchronize()
    return runtime


def prepare_empty_run(engine, batch_sizes, input_shape):
    # allocate buffers
    max_bs = batch_sizes[-1]
    target_bs = max_bs // input_shape[0]
    inputs, outputs, bindings, stream, input_dtype = allocate_buffers(engine, target_bs)
    # prepare input data
    shape = (max_bs,) + input_shape[1:]
    input_data = np.random.randn(*shape)
    np.copyto(inputs[0][0], input_data.ravel())
    return inputs, outputs, bindings, stream


def allocate_buffers(engine, alloc_batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    input_dtype = None

    for tensor in engine:
        size = trt.volume(engine.get_tensor_shape(tensor)) * alloc_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(tensor))
        print('     DATATYPE ', dtype, '\n')
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
            input_dtype = dtype
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream, input_dtype


def free_buffers(inputs, outpus):
    for host_mem, device_mem in inputs:
        host_mem.base.free()
        device_mem.free()
    for host_mem, device_mem in outpus:
        host_mem.base.free()
        device_mem.free()


def prepare_model_engines_and_contexts(model_name, model, batch_sizes, input_shape, use_int8):
    model.to('cuda')
    onnx_file_path_prefix = 'gpu/onnx_models/' + model_name + '_bs'
    engines_contexts = []
    for bs in batch_sizes:
        # export onnx
        onnx_file_path = onnx_file_path_prefix + str(bs) + '.onnx'
        shape = (bs,) + input_shape
        export_onnx(model, onnx_file_path, shape)

        # build engine from onnx file
        engine, context = build_engine(onnx_file_path, use_int8)
        # context.set_binding_shape(engine.get_binding_index('input'), shape)  TODO remove?

        # append engine
        engines_contexts.append((engine, context, shape))

    return engines_contexts


def prepare_engines_and_contexts(model_name, fp_model, q_model, batch_sizes, input_shape, fp_description,
                                 q_description):
    fp_model_name = model_name + '_full_precision'
    q_model_name = model_name + '_int8'

    models = {
        fp_description: prepare_model_engines_and_contexts(fp_model_name, fp_model, batch_sizes, input_shape, False),
        q_description: prepare_model_engines_and_contexts(q_model_name, q_model, batch_sizes, input_shape, True)
    }
    return models


def compute_models_accuracy(models, test_data, logger, seed=0):
    for description, engines_contexts in models.items():

        # find engine with batch size corresponding to target batch_size
        target_batch_size = next(iter(test_data))[0].shape[0]
        packed_engine = find_engine_target_batch_size(engines_contexts, target_batch_size)
        assert packed_engine is not None, f"ERROR, no engine match the test data batch size of {target_batch_size}."

        # allocate cuda memory
        engine, context, input_shape = packed_engine
        inputs, outputs, bindings, stream, input_dtype = allocate_buffers(engine, alloc_batch_size=1)

        # do inference
        correct = 0
        total = 0
        torch.manual_seed(seed)
        for features, labels in test_data:
            features = features.numpy()
            labels = labels.numpy()
            np.copyto(inputs[0][0], features.ravel())
            run(context, inputs, outputs, bindings, stream)
            output = outputs[0][0].reshape(target_batch_size, -1)
            predicted = np.argmax(output, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum()

        # free cuda memory
        free_buffers(inputs, outputs)

        # write accuracy
        accuracy = 100 * correct / total
        logger.write_model_accuracy(description, accuracy)


def compute_models_runtime(models, batch_sizes, logger, nb_executions=20):
    for description, engines_contexts in models.items():
        print('-------> DESCRIPTION     ', description,' <----------------')

        runtimes = []
        for (engine, context, input_shape), bs in zip(engines_contexts, batch_sizes):
            assert input_shape[0] == bs, f"ERROR: the shape of the engine doesn't correspond to the batch size, " \
                                         f"expected {bs} found {input_shape[0]}"

            # allocate cuda memory
            inputs, outputs, bindings, stream, input_dtype = allocate_buffers(engine, alloc_batch_size=1)

            # run
            bs_runtimes = []
            for _ in range(nb_executions):
                input_data = np.random.randn(*input_shape)
                np.copyto(inputs[0][0], input_data.ravel())
                runtime = run(context, inputs, outputs, bindings, stream)
                bs_runtimes.append(runtime * 1000)
            bs_runtimes = np.asarray(bs_runtimes)
            min = np.min(bs_runtimes)
            max = np.max(bs_runtimes)
            median = np.median(bs_runtimes)
            runtimes.append((bs, [min, max, median]))

            # free cuda memory
            free_buffers(inputs, outputs)

        logger.write_model_runtimes(description, runtimes)


def empty_run_models(models, batch_sizes, logger, sleep_interval, run_interval):
    for description, engines_contexts in models.items():
        logger.write_model_empty_run(description)
        for (engine, context, input_shape), bs in zip(engines_contexts, batch_sizes):
            assert input_shape[0] == bs, f"ERROR: the shape of the engine doesn't correspond to the batch size, " \
                                         f"expected {bs} found {input_shape[0]}"

            # prepare empty run
            inputs, outputs, bindings, stream = prepare_empty_run(engine, batch_sizes, input_shape)

            # sleep
            logger.write_model_sleep(bs)
            time.sleep(sleep_interval)

            # run
            logger.write_model_start_run(bs)
            run(context, inputs, outputs, bindings, stream, run_interval)
            free_buffers(inputs, outputs)
            logger.write_model_end_run(bs)
