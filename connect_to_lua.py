import socket

# Connect to BizHawk
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 12345))
print("Connected to BizHawk!")

# Send kick command
sock.send(b"kick\n")
print("Sent: kick")

# Get response
response = sock.recv(1024).decode()
print(f"BizHawk says: {response}")

# Close
sock.close()
print("Done")