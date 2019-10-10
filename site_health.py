#! /usr/bin/env python
"""This script is very useful for when you just to do a health check on a remote server. It does the followings:
  - NSLOOKUP 
  - PING to see if the site is up 
  - Certificate/SSL/TLS info """
import hashlib
import os
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen, ssl, socket


class ServerHealthCheck():

    def __init__(self, base_url, port, tcp):
        self.base_url = base_url
        self.ip_now = self.obtain_ip()
        self.port = port
        self.tcp = tcp
        self.url_path = self.tcp + "://" + base_url
        self.ping_host()
        self.obtain_http_info()
        self.obtain_cert_info()

    def obtain_ip(self):
        print("__LOOKUP____________________________________________")
        currnet_ip = socket.gethostbyname(self.base_url)
        print("ip: " + currnet_ip)
        print("FQDN: " + socket.getfqdn(self.base_url))
        distinct_ips = []
        # 0,0,0,0  is for (family, type, proto, canonname, sockaddr)
        socket_info = socket.getaddrinfo(self.base_url, 0, 0, 0, 0)
        for result in socket_info:
            ns_ip = result[4][0]
            if distinct_ips.count(ns_ip) == 0:
                distinct_ips.append(ns_ip)
                print(ns_ip)
        distinct_ips = list(set(distinct_ips))
        return currnet_ip

    def ping_host(self):
        # ping reesult
        print("\n\n" + "__PING INFO____________________________________________")
        response = os.system("ping -c 1 " + self.ip_now)
        # and then check the response...
        if response == 0:
            print("server " + self.base_url + ": is up ")
        else:
            print("server " + self.base_url + ": is DOWN !!!")

    def obtain_http_info(self):
        print("__SSL/TLS INFO____________________________________________")
        req = Request(self.url_path)
        try:
            response = urlopen(req, context=ssl._create_unverified_context())
            # htmlSource = response.read()
        except HTTPError as e:
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        else:
            print("http code:" + str(response.getcode()))

    def obtain_cert_info(self):
        context = ssl.create_default_context()
        with socket.create_connection((self.base_url, self.port)) as socket_connection:
            with context.wrap_socket(socket_connection, server_hostname=self.base_url) as server_socket:
                # uncomment to print everything
                # print(json.dumps(server_socket.getpeercert() , indent=2, sort_keys=True))
                cert_info = server_socket.getpeercert()
                subject = dict(x[0] for x in cert_info['subject'])
                issued_to = subject['commonName']
                issuer = dict(x[0] for x in cert_info['issuer'])
                issued_by = issuer['commonName']
                valid_from = cert_info['notBefore']
                valid_to = cert_info['notAfter']
                serial_number = cert_info['serialNumber']
                der_cert = server_socket.getpeercert(False)
                der_cert_bin = server_socket.getpeercert(True)
                pem_cert = ssl.DER_cert_to_PEM_cert(server_socket.getpeercert(True))
                # uncomment the below line if you want to see the actual public cert
                # print("certificate pub:",pem_cert)
                thumb_md5 = hashlib.md5(der_cert_bin).hexdigest()
                thumb_sha1 = hashlib.sha1(der_cert_bin).hexdigest()
                thumb_sha256 = hashlib.sha256(der_cert_bin).hexdigest()
                print("issued_to: " + issued_to)
                print("issued_by: " + issued_by)
                print("valid_from: " + valid_from)
                print("valid_to: " + valid_from)
                print("MD5: " + thumb_md5)
                print("SHA1: " + thumb_sha1)
                print("SHA256: " + thumb_sha256)
                print("cipher: " + str(server_socket.cipher()))
                print("SSL/TLS version:  " + server_socket.version())
                print("serial_number: " + serial_number)
                # print(server_socket.shared_ciphers())
            server_socket.close()


if __name__ == '__main__':
    # DO NOT USE IP

    host_name = input("host name ? (example github.com) \n")

    prt = input("port ? \n")
    tcp_it = "https"
    serverHealthCheck = ServerHealthCheck(host_name, prt, tcp_it)
