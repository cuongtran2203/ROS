import torch 
from torch2trt import torch2trt
from torch2trt_dynamic import torch2trt_dynamic
import os
from torch2trt_dynamic import TRTModule
import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()          
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
               
    def __call__(self,x:np.ndarray,batch_size=2):
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]

class ConvertTRT():
    def __init__(self,model_path,net) -> None:
        self.model_path =model_path
        self.model=net.load_state_dict(torch.load(self.model_path)).cuda().eval()
    def convert_TRT(self,percision,width,height,min1=0,min2=0,opt1=0,opt2=0,max1=0,max2=0,dynamic=False):
        dir="weights"
        os.makedirs(dir,exist_ok=True)
        x = torch.ones((1, 3, width, height)).cuda()
        if dynamic:
            # convert to TensorRT feeding sample data as input
            opt_shape_param = [
                [
                    [1, 3, min1, min2],   # min
                    [1, 3, opt1, opt2],   # opt
                    [1, 3, max1, max2]    # max
                ]
            ]
            if percision:
                self.model.half()
                x.half()
                model_trt = torch2trt_dynamic(self.model, [x], fp16_mode=percision, opt_shape_param=opt_shape_param)
            else:
                model_trt = torch2trt_dynamic(self.model, [x], fp16_mode=percision, opt_shape_param=opt_shape_param)
            #export to engine file 
            with open(dir+"/"+"model_trt.engine","wb") as f:
                f.write(model_trt.engine.serialize())
                f.close()
            print("Convert model successefully")
            torch.save(os.path.join(dir,"model_trt.pth"),model_trt)
        else:
            if percision:
                self.model.half()
                x.half()
                model_trt = torch2trt(self.model, [x], fp16_mode=percision)
            else:
                model_trt = torch2trt(self.model, [x], fp16_mode=percision)
            #export to engine file 
            with open(os.path.join(dir,"model_trt.engine"),"wb") as f:
                f.write(model_trt.engine.serialize())
                f.close()
            print("Convert model successefully")
            torch.save(os.path.join(dir,"model_trt.pth"),model_trt)
    def load_modelTRT(self,model_trt,dynamic=False):
        if dynamic:
            model_trt=TRTModule()
            model_trt.load_state_dict(torch.load(model_trt))
        else:
            #load model engine file 
            model = TrtModel(model_trt)
            # batch_size=1
            # data = np.random.randint(0,255,(batch_size,*shape[1:]))/255
            # result = model(data,batch_size)
            
        
        
            
    