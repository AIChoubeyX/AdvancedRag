"""Small fallback for environments that block the native xxhash extension."""

import hashlib


class _Xxh3:
    def __init__(self, data: bytes = b"") -> None:
        self._hash = hashlib.blake2b(data, digest_size=16)

    def digest(self) -> bytes:
        return self._hash.digest()


def xxh3_128(data: bytes = b"") -> _Xxh3:
    return _Xxh3(data)


def xxh3_128_hexdigest(data: bytes = b"") -> str:
    return xxh3_128(data).digest().hex()