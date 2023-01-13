import train
import test

method = "gpnet"
dataset = "AAF" 
model = "Conv3" 
kernel_type = "rbf"

print(f"_____-----{kernel_type}-----_____")   

train.main(method, dataset, model, kernel_type)
result = test.main(method, dataset, model, kernel_type)

print(f"{method} with {dataset} dataset ( with {kernel_type}) : {result}")