"""Minimal public-web HTTP CONNECT proxy for an SSH reverse tunnel.

This is an orchestration utility for GPU hosts without outbound routing.  Bind
it only to loopback, and expose it on the GPU host only through an SSH remote
forward bound to that host's loopback interface.
"""
from __future__ import annotations

import argparse
import ipaddress
import select
import socket
import socketserver
from urllib.parse import urlsplit


def _public_addresses(host: str, port: int) -> list[tuple]:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("proxy target did not resolve")
    for item in addresses:
        value = ipaddress.ip_address(item[4][0])
        if (
            value.is_private
            or value.is_loopback
            or value.is_link_local
            or value.is_multicast
            or value.is_reserved
            or value.is_unspecified
        ):
            raise ValueError("proxy target resolves to a non-public address")
    return addresses


def _connect(host: str, port: int) -> socket.socket:
    last: Exception | None = None
    for family, socktype, proto, _, sockaddr in _public_addresses(host, port):
        try:
            connection = socket.socket(family, socktype, proto)
            connection.settimeout(20)
            connection.connect(sockaddr)
            connection.settimeout(None)
            return connection
        except OSError as error:
            last = error
    raise OSError(f"could not connect to public proxy target: {last}")


def _relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    while sockets:
        readable, _, exceptional = select.select(sockets, [], sockets, 60)
        if exceptional or not readable:
            return
        for source in readable:
            try:
                data = source.recv(65536)
            except OSError:
                return
            if not data:
                return
            target = right if source is left else left
            target.sendall(data)


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(20)
        header = bytearray()
        while b"\r\n\r\n" not in header and len(header) < 65536:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            header.extend(chunk)
        if b"\r\n\r\n" not in header:
            return
        head, remainder = bytes(header).split(b"\r\n\r\n", 1)
        lines = head.split(b"\r\n")
        method, target, version = lines[0].decode("latin-1").split(" ", 2)
        if method.upper() == "CONNECT":
            host, port_text = target.rsplit(":", 1)
            upstream = _connect(host.strip("[]"), int(port_text))
            try:
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if remainder:
                    upstream.sendall(remainder)
                self.request.settimeout(None)
                _relay(self.request, upstream)
            finally:
                upstream.close()
            return
        parsed = urlsplit(target)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            return
        if parsed.scheme == "https":
            self.request.sendall(b"HTTP/1.1 400 Use CONNECT\r\nConnection: close\r\n\r\n")
            return
        port = parsed.port or 80
        upstream = _connect(parsed.hostname, port)
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            rewritten = [f"{method} {path} {version}".encode("latin-1")]
            rewritten.extend(
                line for line in lines[1:]
                if not line.lower().startswith((b"proxy-connection:", b"connection:"))
            )
            rewritten.append(b"Connection: close")
            upstream.sendall(b"\r\n".join(rewritten) + b"\r\n\r\n" + remainder)
            self.request.settimeout(None)
            _relay(self.request, upstream)
        finally:
            upstream.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "::1"}:
        raise SystemExit("the experiment proxy may bind only to loopback")
    with Server((args.bind, args.port), Handler) as server:
        print(f"proxy_ready={args.bind}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
