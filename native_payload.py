"""Bounded NT text extraction from observed field paths; no recipient guessing.

40800 -> 45101 carries body text; 47423 is a cached quote subtree, not a
verified message ID or sender. Independently implemented from documented field
behavior and synthetic tests in the local PriChat generic parser. No data import.
"""
MAX_BYTES = 8192
MAX_FIELDS = 256
MAX_DEPTH = 8


def _varint(data, offset):
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise ValueError('truncated')
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise ValueError('overflow')
        value |= (byte & 127) << (index * 7)
        if not byte & 128:
            return value, offset
    raise ValueError('overflow')


def decode_native_text(data):
    result = dict(text='', native_quote_present=False, native_text_status='unrecognized')
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        return result
    pieces = []
    count = 0

    def walk(packet, depth=0):
        nonlocal count
        if depth > MAX_DEPTH:
            raise ValueError('depth')
        offset = 0
        while offset < len(packet):
            count += 1
            if count > MAX_FIELDS:
                raise ValueError('fields')
            key, offset = _varint(packet, offset)
            field, wire = key >> 3, key & 7
            if not 0 < field < (1 << 29):
                raise ValueError('field')
            if wire == 0:
                _, offset = _varint(packet, offset)
                continue
            if wire in (1, 5):
                offset += 8 if wire == 1 else 4
                if offset > len(packet):
                    raise ValueError('fixed width')
                continue
            if wire != 2:
                raise ValueError('unsupported wire')
            size, offset = _varint(packet, offset)
            end = offset + size
            if end > len(packet):
                raise ValueError('length')
            value = packet[offset:end]
            offset = end
            if field == 47423:
                result['native_quote_present'] = True
                # Never use cached quote text as the current sender's body.
            elif field == 40800:
                walk(value, depth + 1)
            elif field == 45101:
                text = value.decode('utf-8')
                if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                    raise ValueError('control character')
                pieces.append(text)
            # Other fields are opaque; no recursive longest-string heuristic.

    try:
        walk(data)
    except (ValueError, UnicodeError):
        return dict(text='', native_quote_present=False, native_text_status='invalid')
    result['text'] = '\n'.join(pieces)
    result['native_text_status'] = 'parsed' if pieces else 'no_body'
    return result
