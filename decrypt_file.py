from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Protocol.KDF import PBKDF2
import getpass

input_file = 'ipfs_download'

# Generate the key
salt = getpass.getpass(prompt='UserId: ', stream=None) # Salt used as UserId
print('\n');
password = getpass.getpass(prompt='Decryption Password: ', stream=None)  
key = PBKDF2(password, salt, dkLen=32) 

# Read the data from the file
file_in = open(input_file, 'rb') # Open the file to read bytes
iv = file_in.read(16) 
ciphered_data = file_in.read() 
file_in.close()

cipher = AES.new(key, AES.MODE_CBC, iv=iv)  # Setup cipher
original_data = unpad(cipher.decrypt(ciphered_data), AES.block_size) # Decrypt and then up-pad the result
file_out = open("download_zip", "wb") 
file_out.write(original_data)
file_out.close()