## Import all libraries here##
from socket import *
from struct import pack,unpack
import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np

########################################################
## Client Protocol class with methods for connecting, ##
## listening,encoding, decoding, sending and          ##
## receiving Image                                    ##
########################################################
class ClientProtocol:

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
        
    ## Method to encode Image into base64 string ##
    def encode_toString(self,img):
        retval, buffer = cv2.imencode('.jpg',img)
        data = base64.b64encode(buffer)
        return data

    def decode_toImage(self,data):
    
        data_ = base64.b64decode(data)
        img = np.frombuffer(data_, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        return img

    def imshow(self,image, title=None):

        plt.imshow(image)
        if title:
          plt.title(title)

            
    def close(self):
        self.socket.close()
        self.socket = None
        
if __name__ == '__main__':
    
    ## Encode and Send Content Image
    cp = ClientProtocol()
    content_image = cv2.imread('content.jpg')
    image_data = cp.encode_toString(content_image)
    assert(len(image_data))
    cp.connect('127.0.0.1', 55555)
    cp.send_image(image_data)
    cp.close()
    
    ## Receive Stylized Image
    cp = ClientProtocol()
    cp.listen('127.0.0.1', 12344)
    print("Waiting to receive")   
    out_img = cp.receive_images()
    
    ## Decode the output and save it
    out_img =cp.decode_toImage(out_img)
    plt.subplot(1, 2, 1)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    cp.imshow(content_image, 'Content Image')

    plt.subplot(1,2, 2)
    cp.imshow(out_img, 'Style Image')
    plt.imsave("Stylized_Img",out_img)
    print("Received processed Image ")
    cp.close()
   
    
