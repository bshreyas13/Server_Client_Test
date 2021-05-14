
from socket import *
from struct import unpack,pack
import cv2
import Stylize_Image
import base64


class ServerProtocol:

    def __init__(self):
        self.socket = None


    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.settimeout(None)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(5)
    
    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
        
    def receive_images(self):
        
            while True:
                (connection, addr) = self.socket.accept()
                try:
                    bs = connection.recv(8)
                    (length,) = unpack('>Q', bs)
                    data = b''
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += connection.recv(
                            4096 if to_read > 4096 else to_read)

                    # send our 0 ack
                    assert len(b'\00') == 1
                    connection.sendall(b'\00')
                finally:
                    connection.shutdown(SHUT_WR)
                    connection.close()

                return data 
      
        
    def send_image(self, image_data):

        # use struct to make sure we have a consistent endianness on the length
        length = pack('>Q', len(image_data))

        # sendall to make sure it blocks if there's back-pressure on the socket
        self.socket.sendall(length)
        self.socket.sendall(image_data)

    def encode_toString(self,img):
        retval, buffer = cv2.imencode('.jpg',img)
        data = base64.b64encode(buffer)
        return data
            
    def close(self):
        self.socket.close()
        self.socket = None

        # could handle a bad ack here, but we'll assume it's fine.

if __name__ == '__main__':
    
    try:
        sp = ServerProtocol()
        sp.listen('127.0.0.1', 55555)
        print("Socket intialized and listening")
        content_image=sp.receive_images()
        sp.close()
    
        stylized = Stylize_Image.predict(content_image)
    
        print("Stylized Image Obtained")
        sp = ServerProtocol()
        sp.connect('127.0.0.1', 12344)
        sp.send_image(stylized)
        print("Sent Image to CLient")
        sp.close()
    except KeyboardInterrupt:
        sp.close()