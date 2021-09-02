import socket


class ServerManager(object):
    server_root = {
        '219.223.189.131': '/home/zengziyun',
        '219.223.189.198': '/data1/zengziyun',
        '219.223.190.150': '/data/zengziyun',
        '10.103.11.101': '/data/zengziyun'
    }

    @staticmethod
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        return ip

    def get_root(self, ip=None):
        if ip is None:
            ip = self.get_ip()
        root = self.server_root.get(ip)
        if root is None:
            raise RuntimeError('invalid server ip!')
        return root
