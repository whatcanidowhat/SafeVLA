"""Shared GET-only transport/output. No Claim, Git writes, or runtime imports."""
import argparse
import base64
import hashlib
import json
import math
import os
import queue
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'whatcanidowhat/SafeVLA'
API = 'https://api.github.com/repos/' + REPO
SHA = re.compile(r'[0-9a-f]{40}')
CODES = {'SUCCESS': 0, 'OBSERVED': 0, 'BLOCKED': 2, 'UNKNOWN': 2,
         'REMOTE_UNVERIFIED': 2, 'FAILURE': 3, 'CANCELLED': 4,
         'TIMED_OUT': 5, 'NETWORK_ERROR': 6, 'NOT_FOUND': 7,
         'RATE_LIMITED': 8, 'PERMISSION_DENIED': 9, 'INVALID_INPUT': 64}


class Stop(Exception):
    def __init__(self, verdict, reason):
        self.verdict, self.reason = verdict, reason


def require(condition, reason):
    if not condition:
        raise Stop('UNKNOWN', reason)


def report(verdict, evidence=None, blocker=None, next_action='STOP; report to PI'):
    return {'VERDICT': verdict, 'EVIDENCE': {**(evidence or {}), 'CAN_EXECUTE': False},
            'BLOCKER': blocker, 'NEXT_AUTHORIZED_ACTION': next_action}


def emit(result):
    encoded = json.dumps(result, ensure_ascii=True, separators=(',', ':'))
    if len(encoded.encode('utf-8')) >= 2048:
        result = report('UNKNOWN', blocker='OUTPUT_EXCEEDS_2KB; EVIDENCE_WITHHELD')
        encoded = json.dumps(result, separators=(',', ':'))
    print(encoded)
    return CODES[result['VERDICT']]


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # Never echo user arguments (which might contain credentials).
        raise Stop('INVALID_INPUT', 'INVALID_ARGUMENTS')


def positive(value):
    try:
        n = float(value)
        if not math.isfinite(n) or n <= 0:
            raise ValueError
        return n
    except ValueError:
        raise argparse.ArgumentTypeError('positive finite number required')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Stop('NETWORK_ERROR', 'REDIRECT_REFUSED')


class GitHub:
    """Fixed HTTPS origin; GET only; default CA verification; no transport retries.

    A daemon transport thread bounds even a stalled platform DNS/TLS operation.
    Timeout returns once, stops the caller, and never schedules another request.
    """
    def __init__(self, timeout=60, clock=time.monotonic, opener=None):
        self.clock = clock
        self.deadline = clock() + timeout
        self.opener = opener or urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.calls = 0

    def get(self, path):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise Stop('TIMED_OUT', 'DEADLINE_EXCEEDED')
        results = queue.Queue(maxsize=1)
        def request():
            try:
                results.put((True, self._get(path)))
            except Exception as error:
                results.put((False, error))
        threading.Thread(target=request, daemon=True, name='readonly-https').start()
        try:
            ok, value = results.get(timeout=min(10, remaining))
        except queue.Empty:
            if remaining <= 10:
                raise Stop('TIMED_OUT', 'DEADLINE_EXCEEDED') from None
            raise Stop('NETWORK_ERROR', 'REQUEST_TIMEOUT') from None
        if self.clock() >= self.deadline:
            raise Stop('TIMED_OUT', 'DEADLINE_EXCEEDED')
        if not ok:
            raise value
        return value

    def _get(self, path):
        require(path.startswith('/') and not path.startswith('//'), 'INVALID_API_PATH')
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise Stop('TIMED_OUT', 'DEADLINE_EXCEEDED')
        headers = {'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'safevla-readonly',
                   'Cache-Control': 'no-cache'}
        token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
        if token:
            headers['Authorization'] = 'Bearer ' + token
        req = urllib.request.Request(API + path, headers=headers, method='GET')
        self.calls += 1
        try:
            with self.opener.open(req, timeout=min(10, remaining)) as response:
                require(response.geturl() == API + path, 'UNEXPECTED_RESPONSE_ORIGIN')
                chunks, size = [], 0
                while True:
                    if self.clock() >= self.deadline:
                        raise Stop('TIMED_OUT', 'DEADLINE_EXCEEDED')
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    require(size <= 8 * 1024 * 1024, 'API_RESPONSE_TOO_LARGE')
                    chunks.append(chunk)
                value = json.loads(b''.join(chunks))
                require(isinstance(value, dict), 'INVALID_API_RESPONSE')
                return value
        except urllib.error.HTTPError as e:
            if e.code == 429 or (e.code == 403 and
                    (e.headers.get('X-RateLimit-Remaining') == '0' or e.headers.get('Retry-After'))):
                raise Stop('RATE_LIMITED', 'API_RATE_LIMIT') from None
            if e.code in (401, 403):
                raise Stop('PERMISSION_DENIED', 'API_AUTH_OR_PERMISSION') from None
            if e.code == 404:
                raise Stop('NOT_FOUND', 'API_NOT_FOUND_OR_NOT_VISIBLE') from None
            raise Stop('NETWORK_ERROR', 'HTTP_ERROR') from None
        except (urllib.error.URLError, OSError, TimeoutError):
            raise Stop('NETWORK_ERROR', 'HTTPS_DNS_TLS_PROXY_OR_TIMEOUT') from None
        except (ValueError, UnicodeError):
            raise Stop('UNKNOWN', 'INVALID_API_JSON') from None

    def head(self, branch):
        value = self.get('/git/ref/heads/' + urllib.parse.quote(branch, safe=''))
        require(value.get('ref') == 'refs/heads/' + branch, 'REF_MISMATCH')
        obj = value.get('object', {})
        sha = obj.get('sha', '')
        require(obj.get('type') == 'commit' and isinstance(sha, str) and SHA.fullmatch(sha),
                'INVALID_REMOTE_HEAD')
        return sha

    def file(self, revision, path):
        require(bool(SHA.fullmatch(revision)), 'INVALID_REVISION')
        value = self.get('/contents/' + urllib.parse.quote(path, safe='/') + '?ref=' + revision)
        require(value.get('type') == 'file' and value.get('path') == path and
                value.get('encoding') == 'base64', 'INVALID_FILE_RESPONSE')
        try:
            raw = base64.b64decode(''.join(value['content'].split()), validate=True)
        except (ValueError, KeyError, TypeError):
            raise Stop('UNKNOWN', 'INVALID_FILE_ENCODING') from None
        require(len(raw) <= 1024 * 1024 and value.get('size') == len(raw), 'FILE_SIZE_MISMATCH')
        digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(value.get('sha') == digest, 'FILE_BLOB_MISMATCH')
        return raw


def guarded(call):
    try:
        return call()
    except Stop as e:
        return report(e.verdict, blocker=e.reason)
    except KeyboardInterrupt:
        return report('CANCELLED', blocker='INTERRUPTED')
    except Exception:
        # No raw error, URL, token, response body or environment in output.
        return report('UNKNOWN', blocker='INVALID_OR_INCOMPLETE_EVIDENCE')
