import train
import test
import pickle

method = "gpnet"
dataset = "AAF" 
model = "Conv3" 
kernel_type = "gencheb"

train.main(method, dataset, model, kernel_type)
result = test.main(method, dataset, model, kernel_type)

print(f"{method} with {dataset} dataset ( with {kernel_type}) : {result}")