import train
import test
import pickle
import time

method = "gpnet"
dataset = "QMUL" 
model = "Conv3" 
kernel_type = "preio"


errors = []

optim = "SGD"

for kernel_type in ["linear", "rbf", "matern", "poli1", "poli2", "rq", "preio"]:
    results = []
    for method in ["DKT","gpnet"]:
        for dataset in ["QMUL", "AAF"]:
            
            #["cylin", "piece", "rq", "specdel", "add", "rq", "spectral"]:
            try:
                print("----------------------------------------------")
                print(f"{method} with {dataset} dataset with {kernel_type} :")
                print("----------------------------------------------")
                str_time = time.time()
                train.main(method, dataset, model, kernel_type)
                train_end = time.time()
                result = test.main(method, dataset, model, kernel_type)
                test_end = time.time()
                results.append([f"{method} with {dataset} dataset ( with {kernel_type}) : {result}", str_time, train_end, test_end])

                print("----------------------------------------------")
                print(f"{method} with {dataset} dataset ( with {kernel_type}) : {result}")
                print("----------------------------------------------")
            except Exception as e:
                print(f"ERROR : {method} with {dataset} dataset with {kernel_type} :")
                errors.append([f"{method} with {dataset} dataset with {kernel_type} :", e])
    with open(f"src/pickles/results_{optim}_{kernel_type}.pickle","wb") as f:
        pickle.dump(results, f)


print(f"Oprimazer : {optim}")
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


with open("src/pickles/errors.pickle","wb") as f:
    pickle.dump(errors, f)