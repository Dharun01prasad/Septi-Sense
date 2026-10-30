"""
Phone alerts by SMS through an Android phone running "SMS Gateway for Android" (https://sms-gate.app) in
Local Server mode. The laptop and the phone must be on the same Wi-Fi / hotspot; no internet is needed.

  App -> Local Server -> start. It shows   http://<phone-ip>:8080   plus a username and password.
  Put those in the dashboard: Settings -> Phone alerts.

API (local mode): POST http://<phone-ip>:8080/message, Basic auth,
  {"textMessage": {"text": "..."}, "phoneNumbers": ["+91..."]}
"""
import base64, json, re, urllib.error, urllib.request


def normalize_phone(p, country_code='+91'):
    p = re.sub(r'[^\d+]', '', p or '')
    if not p:
        return ''
    if p.startswith('00'):
        p = '+' + p[2:]
    if not p.startswith('+'):
        p = country_code + p.lstrip('0')
    return p


def alarm_text(name, message, demo=False):
    head = 'SeptiSense DEMO (recorded patient replay)' if demo else 'SeptiSense alert'
    return (f'{head} for {name}: {message}. Research prototype, not a diagnosis. '
            'If you feel unwell contact a doctor; emergency 112.')


def send_sms(cfg, phone, text, timeout=8):
    """Returns (ok, detail). Never raises."""
    url = (cfg.get('sms_gateway_url') or '').strip().rstrip('/')
    if not url:
        return False, 'no gateway URL set'
    if not url.startswith('http'):
        url = 'http://' + url
    if not url.endswith('/message'):
        url += '/message'
    body = json.dumps({'textMessage': {'text': text}, 'phoneNumbers': [phone]}).encode()
    req = urllib.request.Request(url, data=body, method='POST', headers={'Content-Type': 'application/json'})
    if cfg.get('sms_user'):
        tok = base64.b64encode(f"{cfg['sms_user']}:{cfg.get('sms_pass', '')}".encode()).decode()
        req.add_header('Authorization', 'Basic ' + tok)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300, f'gateway HTTP {r.status}'
    except urllib.error.HTTPError as e:
        return False, f'gateway HTTP {e.code}: {e.read()[:200].decode(errors="replace")}'
    except Exception as e:
        return False, f'gateway unreachable: {e}'
