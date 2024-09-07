from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
import os

def encrypt_file(file_path, pin):
    backend = default_backend()
    salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=backend
    )
    
    key = kdf.derive(pin.encode())
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=backend)
    encryptor = cipher.encryptor()
    
    with open(file_path, 'rb') as f:
        file_data = f.read()
        
    encrypted_data = encryptor.update(file_data) + encryptor.finalize()
    
    with open(file_path + '.enc', 'wb') as f:
        f.write(salt + iv + encrypted_data)
        
    print(f"File encrypted successfully and saved as {file_path + '.enc'}")

file_path = 'caffe.py'
pin = '1234'
encrypt_file(file_path, pin)
