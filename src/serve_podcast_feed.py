# serve_audio.py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

os.chdir('./podcast_audio')
server = HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
print("Serving audio at http://localhost:8080")
print("Add this to your podcast app: http://localhost:8080/feed.xml")
server.serve_forever()
