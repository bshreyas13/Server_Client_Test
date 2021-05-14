# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import StylizeImage_pb2 as StylizeImage__pb2


class StylizeImageStub(object):
    """service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Stylized_Image = channel.unary_unary(
                '/StylizeImage/Stylized_Image',
                request_serializer=StylizeImage__pb2.B64Image.SerializeToString,
                response_deserializer=StylizeImage__pb2.out_img.FromString,
                )


class StylizeImageServicer(object):
    """service
    """

    def Stylized_Image(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_StylizeImageServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Stylized_Image': grpc.unary_unary_rpc_method_handler(
                    servicer.Stylized_Image,
                    request_deserializer=StylizeImage__pb2.B64Image.FromString,
                    response_serializer=StylizeImage__pb2.out_img.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'StylizeImage', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class StylizeImage(object):
    """service
    """

    @staticmethod
    def Stylized_Image(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/StylizeImage/Stylized_Image',
            StylizeImage__pb2.B64Image.SerializeToString,
            StylizeImage__pb2.out_img.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
