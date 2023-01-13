import train
import test

results = []
kernels = []
get_error = []
for kernel_type in ['gencheb']:#["linear", "rbf", "matern", "poli1", "poli2"]:
   # try:
    print(f"_____-----{kernel_type}-----_____")   

    train.main("DKT","AAF", "Conv3", "gencheb")
    gpnet_aaf_result = test.main("DKT","AAF", "Conv3", "gencheb")

print(f"GPNet with AAF dataset ( with {kernel_type}) : {gpnet_aaf_result}")