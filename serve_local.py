import http.server
import socketserver
import os

PORT = 8000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.startswith('/static/'):
            # Rewrite /static/ to /web/
            path = '/web/' + path[8:]
        elif path == '/':
            path = '/web/index.html'
        else:
            path = '/web' + path

        return super().translate_path(path)

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
