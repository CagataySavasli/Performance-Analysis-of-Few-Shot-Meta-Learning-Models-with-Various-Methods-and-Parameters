import train
import test

results = []
kernels = []
get_error = []
for kernel_type in ["linear", "rbf", "matern", "poli1", "poli2", "cossim", "bncossim"]:
    try:
        print(f"_____-----{kernel_type}-----_____")   

        train.main("gpnet","AAF", "Conv3", kernel_type)
        gpnet_aaf_result = test.main("gpnet","AAF", "Conv3", kernel_type)

        train.main("gpnet","QMUL", "Conv3", kernel_type)
        gpnet_qmul_result = test.main("gpnet","QMUL", "Conv3", kernel_type)

        train.main("DKT","AAF", "Conv3",kernel_type)
        dkt_aaf_result = test.main("DKT","AAF", "Conv3",kernel_type)

        train.main("DKT","QMUL", "Conv3",kernel_type)
        dkt_qmul_result = test.main("DKT","QMUL", "Conv3",kernel_type)

        kernels.append(kernel_type)
        results.append([kernel_type, gpnet_aaf_result, gpnet_qmul_result, dkt_aaf_result, dkt_qmul_result])

    except:
        get_error.append(kernel_type)
for result in results:
    print(f"""
    ******************************************************************************************************
    GPNet with AAF dataset ( with {result[0]}) : {result[1]}
    GPNet with QMUL dataset ( with {result[0]}) : {result[2]}

    DKT with AAF dataset ( with {result[0]}) : {result[3]}
    DKT with QMUL dataset ( with {result[0]}) : {result[4]}
    ******************************************************************************************************
    """)

print("Selected kernels : ")
print(kernels)
print("******************************************************************************************************")
print("Kernels with error :")
print(get_error)
print("******************************************************************************************************")