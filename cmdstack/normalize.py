"""Normalize command lines so that lines doing the same thing look the same.

Each command line gets two forms:

  normalized  conservative. Replaces values that change from run to run but say nothing about behavior
              (user names, IPs, GUIDs, SIDs, ports, hashes, hex, long numbers, temp folders) and keeps the
              rest. This is what the embedding model reads.
  template    aggressive. The normalized form with every remaining number, document name, filter value
              and temp-folder name collapsed. Lines with the same template count as the same behavior.

Obfuscation is kept as evidence in separate columns (non-ASCII, Unicode changes, whitespace padding, carets,
base64), because scrubbing it would erase the signal.

The Windows rules are the ones measured in the write-up. The Linux rules are experimental.
"""
from __future__ import annotations

import base64
import ipaddress
import re
import unicodedata

PLATFORMS = ("windows", "linux")

# --- shared: values that vary but carry no behavior -----------------------------------------------------
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_B64_RE = re.compile(r"(?<![\w+/=])[A-Za-z0-9+/]{40,}={0,2}(?![\w+/=])")
_HASH_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-fA-F]{32,}(?![0-9a-zA-Z])")
_GUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# not after "version=" or "version ": .NET assembly names and installers write versions like 4.0.0.0
_IPV4_RE = re.compile(r"(?<![\w.])(?<!(?i:version)[=: ])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_RE = re.compile(r"(?<![\w:.\]])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?:%\w+)?(?![\w:])")
_IP_PORT_RE = re.compile(r"(<[A-Z_]+_IP>):\d{1,5}\b")
_PORT_RE = re.compile(r"(\bport[ =:]+)\d{1,5}\b", re.IGNORECASE)
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{5,}(?!\w)")
# a number that is a whole quoted argument is an id (wermgr "-outproc" "0" "312" "9812" passes process and thread
# ids this way); left alone, every crash report became its own one-host group
_QUOTED_NUMBER_RE = re.compile(r'"\d{3,}"')

# --- Windows -----------------------------------------------------------------------------------------------
_PATH_PREFIX_RE = re.compile(r"\\\\\?\\|\\\?\?\\")                     # \\?\ and \??\
_DEVICE_VOLUME_RE = re.compile(r"\\device\\harddiskvolume\d+")
_UNC_HOST_RE = re.compile(r"(?<![A-Za-z0-9:.\\])\\\\(?!\.\\)[^\\/:*?\"<>|\s]+")
_SID_RE = re.compile(r"\bs-1-5-21(?:-\d+){3,4}\b")
# a user name may contain spaces but not \ / " : ; | = , + * ? < > [ ] &; Public, Default and All Users are kept
_NAME = r"[^\\/\"'\s:;|=,+*?<>\[\]&]+"
_WIN_USER_PATH_RE = re.compile(
    rf"([\\/]users[\\/])(?!(?:public|default|all users)(?:[\\/\"'\s]|$))(?:{_NAME}(?: {_NAME})*(?=[\\/])|{_NAME})"
)
_WIN_TEMP_RE = re.compile(r"c:\\users\\<USER>\\appdata\\local\\temp")
# Windows Search names its pipes after the session or user (...pipe12_, ...pipe_azuread\jsmith52_), which would
# otherwise split the same indexer process into one group per machine
_SEARCH_PIPE_RE = re.compile(r"(global\\usgthr(?:ctrl)?fltpipemssgthrpipe)\S*")
_ENV_VARS = {
    "%systemroot%": r"c:\windows",
    "%windir%": r"c:\windows",
    "%comspec%": r"c:\windows\system32\cmd.exe",
    "%programfiles%": r"c:\program files",
    "%programfiles(x86)%": r"c:\program files (x86)",
    "%programdata%": r"c:\programdata",
    "%userprofile%": r"c:\users\<USER>",
    "%appdata%": r"c:\users\<USER>\appdata\roaming",
    "%localappdata%": r"c:\users\<USER>\appdata\local",
    "%public%": r"c:\users\public",
    "%temp%": "<TEMP>",
    "%tmp%": "<TEMP>",
}

# --- Linux (experimental) ----------------------------------------------------------------------------------
_LX_HOME_RE = re.compile(r"(?<![\w.])/home/[^/\s\"']+")
_LX_TMP_RE = re.compile(r"(?<![\w.])/(?:var/)?tmp(?=/)")
_LX_PROC_PID_RE = re.compile(r"/proc/\d+")
_LX_PID_ARG_RE = re.compile(r"(\b(?:kill|pkill|renice|strace|gdb|lsof)\b[^|;&]*?\s(?:-p\s*)?)\d{2,}\b")
_LX_CONTAINER_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{12}(?![0-9a-f])")

# --- template-only collapsing ------------------------------------------------------------------------------
_DOC_EXT = r"\.(?:pdf|docx?|xlsx?|pptx?|rtf|txt|csv)"
_QUOTED_DOC_RE = re.compile(rf"\"[^\"]*{_DOC_EXT}\"", re.IGNORECASE)
_DOC_RE = re.compile(rf"[^\s\"]+{_DOC_EXT}\b", re.IGNORECASE)
_FILTER_VALUE_RE = re.compile(r"(\beq )[^*\"]+")                       # taskkill/tasklist /fi "... eq <value>"
_TEMP_CHILD_RE = re.compile(r"<TEMP>[\\/][^\\/\s\"]+")
_DIGITS_RE = re.compile(r"(?<![A-Z_])\d+")                            # the guard keeps placeholder tokens intact

_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_BROADCAST = ipaddress.ip_address("255.255.255.255")


def parse_networks(text_or_list) -> list:
    """Turn '10.0.0.0/8,142.20.0.0/16' (or a list) into ip_network objects."""
    if not text_or_list:
        return []
    items = text_or_list.split(",") if isinstance(text_or_list, str) else list(text_or_list)
    return [ipaddress.ip_network(item.strip(), strict=False) for item in items if str(item).strip()]


def _ip_token(text: str, internal_nets) -> str | None:
    """Return a category token for an IP address, or None if the text is not an IP."""
    try:
        address = ipaddress.ip_address(text.split("%")[0])
    except ValueError:
        return None
    if any(address in net for net in internal_nets if net.version == address.version):
        return "<INTERNAL_IP>"
    if address.is_unspecified:
        return "<ANY_IP>"
    if address == _BROADCAST:
        return "<BROADCAST_IP>"
    if address.is_loopback:
        return "<LOOPBACK_IP>"
    if address.is_link_local:                    # before is_private, which also covers 169.254/16
        return "<LINK_LOCAL_IP>"
    if address.is_multicast:
        return "<MULTICAST_IP>"
    if address.is_private or address in _CGNAT:
        return "<PRIVATE_IP>"
    return "<PUBLIC_IP>"


TRUNCATED = " [truncated]"


def _decode_b64(blob: str) -> str | None:
    """Return the decoded text if blob is base64 of readable text (UTF-16LE as used by -enc, or UTF-8).

    Telemetry often cuts long command lines, leaving base64 whose length is not a multiple of four. Then the
    longest decodable prefix is used and the text is marked as truncated, because the readable start of a
    PowerShell stager is the evidence an analyst needs.
    """
    truncated = len(blob) % 4 != 0                # encoders pad complete base64 to a multiple of four
    try:
        data = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
    except ValueError:
        try:
            data = base64.b64decode(blob[: len(blob) - len(blob) % 4], validate=True)
            truncated = True
        except ValueError:
            return None
    for encoding in ("utf-16-le", "utf-8"):
        chunk = data[: len(data) - len(data) % 2] if encoding == "utf-16-le" else data
        try:
            text = chunk.decode(encoding, errors="strict" if not truncated else "ignore")
        except UnicodeDecodeError:
            continue
        readable = sum(c.isascii() and (c.isprintable() or c.isspace()) for c in text)
        if text and readable / len(text) > 0.9:
            return text + (TRUNCATED if truncated or len(chunk) < len(data) else "")
    return None


def _windows_paths(value: str) -> str:
    """Case, path prefix, device volume, UNC host, user, SID and env-var normalization."""
    value = value.lower()
    value = _PATH_PREFIX_RE.sub("", value)
    value = _DEVICE_VOLUME_RE.sub("c:", value)
    value = _UNC_HOST_RE.sub(r"\\\\<HOST>", value)   # after lower(), so <HOST> stays upper-case
    value = _SID_RE.sub("<SID>", value)
    value = _WIN_USER_PATH_RE.sub(r"\1<USER>", value)
    for name, replacement in _ENV_VARS.items():
        value = value.replace(name, replacement)
    value = _SEARCH_PIPE_RE.sub(r"\1<ID>", value)
    return _WIN_TEMP_RE.sub("<TEMP>", value)


def _linux_paths(value: str) -> str:
    """Linux keeps case (paths are case-sensitive); only the varying parts are replaced."""
    value = _LX_HOME_RE.sub("/home/<USER>", value)
    value = _LX_TMP_RE.sub("<TEMP>", value)
    value = _LX_PROC_PID_RE.sub("/proc/<PID>", value)
    value = _LX_PID_ARG_RE.sub(r"\1<PID>", value)
    return _LX_CONTAINER_ID_RE.sub("<CONTAINER>", value)


def normalize(command_line, platform: str = "windows", internal_nets=()) -> tuple[str, dict]:
    """Return (normalized, features) for one command line."""
    raw = "" if command_line is None else str(command_line)
    if raw.lower() == "nan":
        raw = ""
    value = unicodedata.normalize("NFKC", raw)
    features = {
        "non_ascii": any(ord(c) > 127 for c in raw),
        "unicode_changed": value != raw,
        "whitespace_runs": len(re.findall(r"\s{2,}", raw)),
        "caret_count": raw.count("^"),
    }
    value = _CONTROL_RE.sub(" ", value)
    value = _HASH_RE.sub("<HASH>", value)        # before base64: hex is a subset of the base64 alphabet
    decoded = []

    def replace_b64(match):                      # runs before lowercasing, because base64 is case-sensitive
        text = _decode_b64(match.group(0))
        if text is None:
            return "<ENCBLOB>"
        decoded.append(text)
        return "<ENC>"

    value = _B64_RE.sub(replace_b64, value)
    if platform == "windows":
        value = _windows_paths(value)
        value = value.replace("<enc>", "<ENC>").replace("<encblob>", "<ENCBLOB>").replace("<hash>", "<HASH>")
    else:
        value = _linux_paths(value)
    value = _GUID_RE.sub("<GUID>", value)
    value = _IPV4_RE.sub(lambda m: _ip_token(m.group(0), internal_nets) or m.group(0), value)
    value = _IPV6_RE.sub(lambda m: _ip_token(m.group(0), internal_nets) or m.group(0), value)
    value = _IP_PORT_RE.sub(r"\1:<PORT>", value)
    value = _PORT_RE.sub(r"\1<PORT>", value)
    value = _HEX_RE.sub("<HEX>", value)
    value = _LONG_NUMBER_RE.sub("<NUM>", value)
    value = _QUOTED_NUMBER_RE.sub('"<NUM>"', value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    features["b64_decoded"] = " | ".join(decoded)
    return value, features


def template(normalized: str) -> str:
    """Collapse a normalized line into a grouping key: lines with the same template are the same behavior."""
    value = _QUOTED_DOC_RE.sub("<DOC>", normalized)
    value = _DOC_RE.sub("<DOC>", value)
    value = _FILTER_VALUE_RE.sub(r"\1<VALUE>", value)
    value = _TEMP_CHILD_RE.sub(lambda m: m.group(0)[:7] + "<R>", value)   # keeps the separator after <TEMP>
    value = value.replace('"', "")
    value = _DIGITS_RE.sub("<N>", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


_EXECUTABLE_RE = re.compile(r"^.*?\.(?:exe|com|bat|cmd|ps1|msi|scr|dll)(?=[\s\"]|$)", re.IGNORECASE)


def program(normalized: str) -> str:
    """The program a normalized line runs, as a short name (for summaries)."""
    text = normalized.strip()
    if text.startswith('"'):
        end = text.find('"', 1)
        first = text[1:end] if end > 0 else text[1:]
    else:
        # unquoted Windows paths can contain spaces (c:\program files\...), so read up to the executable
        first = text.split(" ", 1)[0]
        match = _EXECUTABLE_RE.match(text) if "\\" in first else None
        if match:
            first = match.group(0)
    return re.split(r"[\\/]", first)[-1] or first
