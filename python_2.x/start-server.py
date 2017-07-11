from __future__ import print_function
import SimpleHTTPServer
import SocketServer

PORT = 8000 #This will serve at port 8080 

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer(("", PORT), Handler)

print("serving at port", PORT)
httpd.serve_forever()
