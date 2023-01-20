import train
import test
import pickle

method = "gpnet"
dataset = "QMUL" 
model = "Conv3" 
kernel_type = "preio"

results = []
errors = []
for method in ["gpnet"]:
    for dataset in ["QMUL"]:
        for kernel_type in ["piece"]: 
            #["cylin", "piece", "rq", "specdel", "add", "rq", "spectral"]:
            train.main(method, dataset, model, kernel_type)
            result = test.main(method, dataset, model, kernel_type)
            try:
                print("----------------------------------------------")
                print(f"{method} with {dataset} dataset with {kernel_type} :")
                print("----------------------------------------------")

                train.main(method, dataset, model, kernel_type)
                result = test.main(method, dataset, model, kernel_type)
                results.append(f"{method} with {dataset} dataset ( with {kernel_type}) : {result}")

                print("----------------------------------------------")
                print(f"{method} with {dataset} dataset ( with {kernel_type}) : {result}")
                print("----------------------------------------------")
            except:
                print(f"ERROR : {method} with {dataset} dataset with {kernel_type} :")
                errors.append(f"{method} with {dataset} dataset with {kernel_type} :")

print("RESULTS : ")
print("----------------------------------------------")
for res in results:
    print(res)
    print("----------------------------------------------")
print("ERRORS : ")
print("************************************************")
for res in errors:
    print(res)
    print("************************************************")

with open("results.pickle","wb") as f:
    pickle.dump(results, f)
with open("errors.pickle","wb") as f:
    pickle.dump(errors, f)