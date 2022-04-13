from tvm.driver import tvmc

target = "llvm -mcpu=znver2"
device = "cpu"
net_model = "resnet50-v1-7.onnx"
do_tuning = False
records_file = "resnet50.log"
compiled_package_path = "./compiled_pckg/compiled_resnet50"

#step 1: load
model = tvmc.load(net_model,shape_dict={})
model.summary()
#step 2: optional tune the model
if do_tuning:
    tvmc.tune(model,target=target,tuning_records=records_file)
else:
    records_file = None        
#step 3: Compile 
package = tvmc.compile(model,target=target,tuning_records=records_file,package_path=compiled_package_path)
tvmc.TVMCpackage(package_path=compiled_package_path)
#step 4: 
result = tvmc.run(package, device=device)
print(result)