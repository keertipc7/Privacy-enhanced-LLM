from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Protocol.KDF import PBKDF2
import getpass

# Generate the key
salt = getpass.getpass(prompt='UserId: ', stream=None) 
print('\n');
password = getpass.getpass(prompt='Encryption Password: ', stream=None)  
key = PBKDF2(password, salt, dkLen=32) # Your key that you can encrypt with

output_file = 'encrypted.bin' # Output file
data_location = "DB_zip.zip"
with open(data_location, 'rb') as file_data:
    data = file_data.read()

# Create cipher object and encrypt the data
cipher = AES.new(key, AES.MODE_CBC) 
ciphered_data = cipher.encrypt(pad(data, AES.block_size)) # Pad the input data and then encrypt

file_out = open(output_file, "wb") # Open file to write bytes
file_out.write(cipher.iv) 
file_out.write(ciphered_data) 
file_out.close()