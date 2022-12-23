import train
import test

train.main("transfer","QMUL", "Conv3")
transfer_qmul_result = test.main("transfer","QMUL", "Conv3")

train.main("transfer","AAF", "Conv3")
transfer_aaf_result = test.main("transfer","AAF", "Conv3")

train.main("gpnet","QMUL", "Conv3")
gpnet_qmul_result = test.main("gpnet","QMUL", "Conv3")

train.main("gpnet","AAF", "Conv3")
gpnet_aaf_result = test.main("gpnet","AAF", "Conv3")

train.main("DKT","QMUL", "Conv3")
dkt_qmul_result = test.main("DKT","QMUL", "Conv3")

train.main("DKT","AAF", "Conv3")
dkt_aaf_result = test.main("DKT","AAF", "Conv3")

print(f"""
******************************************************************************************************
FeatureTransfer with QMUL dataset : {transfer_qmul_result}
FeatureTransfer with AAF dataset : {transfer_aaf_result}
GPNet with QMUL dataset : {gpnet_qmul_result}
GPNet with AAF dataset : {gpnet_aaf_result}
DKT with QMUL dataset : {dkt_qmul_result}
DKT with AAF dataset : {dkt_aaf_result}
******************************************************************************************************
""")