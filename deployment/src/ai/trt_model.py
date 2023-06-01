import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TrtModel(object):

    def __init__(self, onnx_file):
        super().__init__()
        engine = self.build_engine(onnx_file, max_batch_size=5)
        self.context = engine.create_execution_context()
        (
            self.bindings, self.device_inputs, self.input_shapes,
            self.host_outputs, self.device_outputs, self.output_names
        ) = self.allocate_buffers(engine)

        self.stream = cuda.Stream()

    def build_engine(self, onnx_file, max_batch_size):
        trt_file = onnx_file.with_suffix(".trt")
        assert onnx_file.is_file() or trt_file.is_file()
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        if trt_file.is_file():
            with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
        else:
            EXPLICIT_BATCH = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
            with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                    builder.create_builder_config() as config, \
                        trt.OnnxParser(network, TRT_LOGGER) as parser, \
                            trt.Runtime(TRT_LOGGER) as runtime:
                # use up to 1GB of GPU memory for tactic selection
                config.max_workspace_size = 1 << 30
                # config.set_flag(trt.BuilderFlag.FP16)
                builder.max_batch_size = max_batch_size

                with open(onnx_file, "rb") as model:
                    parser.parse(model.read())

                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                # engine = builder.build_engine(network, config)
                # # engine = builder.build_cuda_engine(network)
                # plan = engine.serialize()
                with open(trt_file, "wb") as f:
                    f.write(plan)
        return engine

    def allocate_buffers(self, engine):
        bindings = []
        device_inputs = []
        input_shapes = []
        host_outputs = []
        device_outputs = []
        output_names = []
        for binding in engine:
            dims = engine.get_binding_shape(binding)
            # size = trt.volume(dims) * engine.max_batch_size
            dims = (engine.max_batch_size, *dims[1:])
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(dims, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                device_inputs.append(device_mem)
                input_shapes.append(dims)
            else:
                host_outputs.append(host_mem)
                device_outputs.append(device_mem)
                index = engine.get_binding_index(binding)
                output_names.append(engine.get_binding_name(index))
        return (
            bindings, device_inputs, input_shapes,
            host_outputs, device_outputs, output_names
        )

    def __call__(self, *host_inputs):
        for host_input, input_shape, device_input in zip(
            host_inputs, self.input_shapes, self.device_inputs
            ):
            assert host_input.shape[1:] == input_shape[1:]
            cuda.memcpy_htod_async(device_input, host_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        for device_output, host_output in zip(self.device_outputs, self.host_outputs):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
        self.stream.synchronize()
        batch_size = host_inputs[0].shape[0]
        preds = {
            output_name: host_output[:batch_size, ...]
            for output_name, host_output in zip(self.output_names, self.host_outputs)
        }
        return preds
