# GVON Nullifier: filtering x.com at the network layer

`gvon/nullifier.py` is a [mitmproxy](https://mitmproxy.org) addon. Your browser talks to x.com through a
local proxy; the proxy decrypts the timeline JSON, removes nullified posts with `gvon/prune.py`, and
re-encodes the response. The removed posts never reach the browser, the same way an ad blocker drops
requests before they render.

Why a proxy and not DNS or `/etc/hosts`: x.com is HTTPS and every post comes from the same hosts
(`x.com`, `twitter.com`, `api.x.com`, `api.twitter.com`). DNS blocking can only block all of X or none of
it. Filtering individual posts means reading inside the TLS connection, and that requires a CA your
browser trusts.

What gets filtered: JSON responses with status 200 from those four hosts under `/i/api/graphql/*`,
`/graphql/*`, `/i/api/2/notifications/*` and `/i/api/2/timeline/*`. That covers the home timelines,
replies under a post (TweetDetail), search, profiles, lists and notifications. Cursor entries are never
touched, so infinite scroll keeps working. Your own posts are never judged. The account is identified by
the `twid` cookie and `GVON_HANDLE`.

## 1. Run the proxy

```sh
cd good-vibes-only
.venv/bin/mitmdump -s gvon/nullifier.py            # port 8080, on ALL interfaces by default
```

mitmproxy binds every interface unless you pass `--listen-host 127.0.0.1`. `make proxy` also sets
`block_global=false`, which admits clients from public IP addresses. Anyone who can reach port 8080 can
then use your TLS-intercepting proxy. On café Wi-Fi or any shared network, firewall the port, or run
mitmdump yourself with `--listen-host 127.0.0.1` (or `--proxyauth user:pass`).

The first run creates the mitmproxy CA in `~/.mitmproxy/`. On startup the addon logs the classifier's
device (`mps` on Apple Silicon, otherwise `cpu`) and how long it took to load. Each pruned item gets one
log line, `gvon: pruned HomeTimeline entry=... @user (reason)`. If `models/latest` does not exist yet,
the addon logs `DEGRADED MODE` and filters by `data/blocklist.json` only.

Environment options (or set them in `.env`):

| var | effect |
| --- | --- |
| `GVON_DRY_RUN=1` | log what would be pruned, forward responses unmodified |
| `GVON_RECORD=1` | save every matched response body to `data/recordings/<ts>_<path>.json` (no headers or cookies) so you can study real response *shapes*. Recordings are real timeline data. Keep them under `data/`, and write any fixture by hand (see Privacy). |
| `GVON_THRESHOLD=0.7` | explicit threshold override (also used by `python -m gvon.classifier`). It re-enables text scoring if training disabled it. |
| `GVON_TEXT_SCOPE=all` | score the text of every tweet. The default, `engagements`, scores only tweets that reply to, mention or quote you, and checks every other tweet against the blocklist only. |
| `GVON_WATCH_DELTA=0.15` | how much lower the threshold is for `data/watchlist.json` authors |
| `GVON_HANDLE=yourname` | your account; its posts are never pruned |
| `GVON_MODEL_DIR`, `GVON_BLOCKLIST` | override `models/latest` and `data/blocklist.json` |

When `data/blocklist.json` changes on disk, the addon reloads it and `data/watchlist.json` without a
restart.

To check a saved body offline, run
`.venv/bin/python -m gvon.nullifier data/recordings/<file>.json`
(full classifier) or `.venv/bin/python -m gvon.prune <file>.json` (blocklist only).

## 2. Trust the mitmproxy CA (macOS)

Add the CA to your **login** keychain as a trusted root. This needs no `sudo`, and macOS asks for your
password:

```sh
security add-trusted-cert -r trustRoot -k ~/Library/Keychains/login.keychain-db ~/.mitmproxy/mitmproxy-ca-cert.pem
```

Chrome, Safari and Edge on macOS use the keychain. To remove it later:

```sh
security delete-certificate -c mitmproxy ~/Library/Keychains/login.keychain-db
```

Treat `~/.mitmproxy/mitmproxy-ca.pem` (the CA **private key**) like a password. Anyone who has it can
intercept TLS on any machine that trusts this CA. Never copy it off the machine or commit it. `*.pem` is
gitignored. The key is generated per machine, so each install has its own.

Keychain trust applies to the whole system, but only browsers you point at the proxy are affected.
Everything else connects directly and never sees the mitmproxy certificate.

## 3. Proxy only one browser: a dedicated Chrome profile

Start a separate Chrome instance with its own profile directory and the proxy flag. Your normal Chrome
stays unproxied.

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --user-data-dir="$HOME/.gvon-chrome" \
  --proxy-server="127.0.0.1:8080" \
  --proxy-bypass-list="<-loopback>"
```

Log in to x.com in that window. The `gvon: pruned` lines show up in the mitmdump terminal.

Optional: send only X through the proxy and everything else direct, using a PAC script:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --user-data-dir="$HOME/.gvon-chrome" \
  --proxy-pac-url='data:application/x-javascript-config;base64,'$(printf '%s' 'function FindProxyForURL(u,h){return /(^|\.)(x|twitter)\.com$/.test(h)?"PROXY 127.0.0.1:8080":"DIRECT";}' | base64)
```

## 4. Firefox alternative

Firefox keeps its own certificate store and proxy settings, so it is easy to isolate:

1. `open -a Firefox --args -P` and create a profile named `gvon`.
2. Settings → Network Settings → Manual proxy configuration: HTTP proxy `127.0.0.1`, port `8080`, and
   tick "Also use this proxy for HTTPS".
3. Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import
   `~/.mitmproxy/mitmproxy-ca-cert.pem`, and tick "Trust this CA to identify websites". Alternatively, set
   `security.enterprise_roots.enabled = true` in `about:config` to use the macOS keychain from step 2.

## 5. Other machines, a router, phones

The proxy can serve other devices on your LAN:

- **Explicit proxy:** `.venv/bin/mitmdump -s gvon/nullifier.py --listen-host 0.0.0.0`, then set each
  device's Wi-Fi HTTP proxy to `<this-mac-ip>:8080`. By default mitmproxy refuses clients from public IP
  ranges (`block_global`). Add `--proxyauth user:pass` on shared networks.
- **WireGuard mode (easiest for phones):** `.venv/bin/mitmdump -s gvon/nullifier.py --mode wireguard`,
  then scan the printed QR code with the WireGuard app.
- **Transparent mode (router or gateway box):** `.venv/bin/mitmdump -s gvon/nullifier.py --mode transparent --showhost`
  on a Linux gateway, with iptables/nftables redirecting TCP 80/443 to port 8080 (on macOS, use a `pf`
  `rdr` rule). Block UDP 443 so browsers fall back from QUIC/HTTP3 to TCP, where the proxy can see the traffic.
  See mitmproxy's "transparent proxying" docs for the exact rules.

**Every device needs the CA installed and trusted.** Open `http://mitm.it` through the proxy for
per-OS instructions. On iOS, after installing the profile you must also enable it under Settings → General →
About → Certificate Trust Settings.

**The native X apps (iOS, Android, macOS) pin their certificates.** They reject the mitmproxy CA, so their
traffic cannot be filtered and either passes through unfiltered or fails. Only the x.com website in a
browser is filtered. On a phone, use x.com in the mobile browser.

## 6. No CA? Userscript fallback

`.venv/bin/python scripts/gvon-userscript.py` writes `dist/gvon.user.js` (gitignored, because it contains
blocklisted usernames). Install it in Tampermonkey or Violentmonkey. It hides `article[data-testid=tweet]`
elements whose author link matches the blocklist. It is weaker than the proxy:

- The posts are still downloaded.
- It knows the blocklist only, with no classifier scoring.
- Posts are hidden after they render.

Re-run the script after the blocklist changes.

## Privacy

Nothing leaves your machine:

- TLS is decrypted locally by mitmproxy.
- Decisions come from the local classifier, which runs on CPU or Apple Metal (MPS), and the local
  `data/blocklist.json`.
- The addon makes no network requests of its own and sends no telemetry.

The proxy necessarily handles your X session cookies in memory while forwarding requests, but never
writes them anywhere. `GVON_RECORD=1` saves response bodies (your timeline content) under
`data/recordings/`, which is gitignored. `fixtures/` is tracked by git, so never copy a recording into
it. A fixture must be hand-written synthetic data that only mimics a recording's structure, with invented
text, ids and `@synthetic_*` handles. Delete recordings when you no longer need them.

## Limitations

- X changes its web client often. `gvon/prune.py` is structure-agnostic: it detects tweets by
  `legacy.full_text`/`note_tweet` and entries by `entryId`, not by operation names. Use `GVON_RECORD=1`
  plus the tests to catch shape changes. If pruning throws, the original response is passed through
  untouched.
- A pruned page can show fewer posts than X intended. Scrolling loads more.
- Not covered: DMs, search typeahead, Spaces, and anything delivered over WebSockets or via the native apps.
