import grpc
import asyncio
import json
import time

from service import service_pb2
from service import service_pb2_grpc

from infer import infer

class InferServiceServicer(service_pb2_grpc.InferServiceServicer):
    
    async def StartInfer(self, request, context):
        try:
            response_data = await infer(request.image_path)

            return service_pb2.InferResponse(
                response=json.dumps(response_data), 
                logId=request.logId,
                sessionId=request.sessionId
            )
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            context.set_details(f'Error during inference: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return service_pb2.InferResponse()

async def serve():
    server = grpc.aio.server()
    
    service_pb2_grpc.add_InferServiceServicer_to_server(InferServiceServicer(), server)
    
    server.add_insecure_port('[::]:7777')
    await server.start()
    print("Server started on port 7777")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)

if __name__ == '__main__':
    asyncio.run(serve())
