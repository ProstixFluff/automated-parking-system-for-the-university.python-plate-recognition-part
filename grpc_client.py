import grpc

import LicenseService_pb2
import LicenseService_pb2_grpc

def run():
	with grpc.insecure_channel('localhost:8080') as channel:
		stub = LicenseService_pb2_grpc.LicensePlateServiceStub(channel)

		response = stub.greeting(LicenseService_pb2.LicensePlateRequest(PlateRequest))
run()