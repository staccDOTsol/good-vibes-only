# Telegram

GVON's X filter is a mitmproxy addon that edits x.com's JSON before the browser renders it. Telegram cannot be
filtered that way. Its clients speak MTProto, Telegram's own encrypted protocol, so a TLS proxy sees nothing it
can read or edit, and the official apps would reject any change.

What GVON does instead is run a second Telegram client, logged in as you, the **sink**
(`gvon/telegram_nullifier.py`, Telethon). It receives every incoming message at the same time as your phone and
desktop apps, scores it with the same local classifier and blocklist, and within milliseconds deletes, mutes or
archives it in your account, before you open the chat. This is the closest Telegram gets to "it never reaches
my eyes". It is not a network filter: see [Limits](#limits).

There are two pieces:

| target | module | what it does |
| --- | --- | --- |
| `make telegram-login` | `gvon/telegram_ingest.py` | one-time interactive login (phone number + code, plus 2FA password if set), writes `data/telegram.session` |
| `make telegram-ingest` | `gvon/telegram_ingest.py` | pulls the last 7 days of messages for the teacher to label (the Telegram counterpart of `make ingest`) |
| `make telegram` | `gvon/telegram_nullifier.py` | the sink: runs until stopped and acts on every incoming message |
| `make telegram-vault N=20` | `gvon/telegram_nullifier.py --vault 20` | prints the last N nullified messages so you can audit false positives |

## 1. Get TG_API_ID and TG_API_HASH

Every Telegram client needs an API id/hash pair registered to a Telegram account.

1. Open <https://my.telegram.org> and log in with your phone number. The login code arrives in your Telegram app,
   not by SMS.
2. Choose **API development tools**.
3. Fill in the form. Any app title and short name will do (for example `gvon` / `gvonsink`), platform "Desktop",
   URL and description may be left empty. Submit.
4. Copy **App api_id** (a number) and **App api_hash** (a 32-character hex string) into `.env`:

   ```sh
   TG_API_ID=1234567
   TG_API_HASH=0123456789abcdef0123456789abcdef
   ```

`.env` is gitignored, and GVON never prints or logs these values. The api_hash is not a password to your account,
but it identifies an app registered to you. Do not publish it.

## 2. Log in once

```sh
make telegram-login
```

This asks for your phone number, the login code Telegram sends to your other sessions, and your cloud password if
two-step verification is on. It writes `data/telegram.session`. You only do this once. The sink and ingest reuse
the session and never prompt. If the sink says the session is not logged in, run `make telegram-login` again.

The new session shows up in Telegram under **Settings -> Devices** as an active session. Terminating it there logs
the sink out. That is also the way to revoke it remotely.

## 3. Pull a week for the teacher (optional but recommended)

```sh
make telegram-ingest
```

This pulls the last 7 days of messages so they can be labelled and trained on, as `make ingest` does for X. The
sink works without it, using the classifier trained on your X week. It is more accurate once the student has also
seen Telegram text, and only Telegram verdicts put Telegram senders on the blocklist.

`make all` (and `scripts/run-all.sh`) includes this step once `TG_API_ID`, `TG_API_HASH` and a session are present:
X ingest, then Telegram ingest, then `label` over both sources, then `train`. Without them it prints what is
missing and runs X only.

**One process per session.** The sink, `pull` and `login` each hold an exclusive lock on
`data/telegram.session.lock` (`data/telegram.tg_session.lock` when `TG_SESSION` is used) while they run.
Telethon keeps the session in SQLite and commits only about once a minute, so two processes on one session
fail with "database is locked": the pull dies while connecting, or the sink drops a batch of updates and
never acts on those messages. So while the sink is running, `make telegram-ingest` stops with
"in use by sink (make telegram)", and `make all` / `scripts/run-all.sh` skip the Telegram pull with that
reason and carry on with X. Stop the sink, run `make telegram-ingest`, then start the sink again. The lock is
released when the process exits, even after a crash.

A chat Telegram refuses to read (for example a supergroup you were banned from, `ChannelPrivateError`) no
longer stops the pull: it is logged with its id and error name, counted under `chats_failed` in
`data/raw/_telegram_summary.json`, and retried on the next run. `ChannelForbidden` / `ChatForbidden` dialogs are
skipped up front (`chats_skipped.forbidden`).

## 4. Run the sink

```sh
make telegram                                        # run until Ctrl-C
make telegram TG_ARGS="--dry-run"                    # decide + vault only, change nothing in Telegram
make telegram TG_ARGS="--backfill 50"                # also score the last 50 unread messages per chat at start
```

At start-up the sink loads `gvon.classifier.Nullifier` once and logs the device (`mps` or `cpu`) and the load time
in ms. If there is no trained model, it logs `DEGRADED MODE` and acts on the blocklist only. Then it handles
`NewMessage(incoming=True)` events. For each incoming message that has text:

- **Exempt senders.** Telegram's service account `777000` (login codes, "new login" and security alerts),
  Telegram support accounts (the `support` flag) and every sender in `GVON_TG_ALLOW` are never scored or acted
  on. The ingest never collects `777000` either, so the model has never seen that text.
- **Decision.** The message is nullified if the sender is on the blocklist (by Telegram user id or @username), or
  if `Nullifier.should_nullify` says so: score >= threshold, with the lower bar for watch-listed senders and the
  per-sender running-mean rule, exactly as on X. Messages without text (photos without captions, stickers,
  service messages) are ignored, and so are your own messages. The classifier runs in a worker thread, so the
  model never blocks Telethon's network loop, and the score recorded in the vault is computed only for
  nullified messages (not at all for a blocklist hit while text scoring is off; the vault then shows `null`).
- **Sender context.** When the installed student was trained with author inputs (`student_inputs` in
  `models/latest/config.json` lists `author_bio` / `author_stats`), the sink also passes the sender's verified
  flag and bio. The bio costs one `GetFullUserRequest` per sender. A sender the event did not resolve costs one
  `get_entity` call. Both results are cached per sender id for a day, so the calls never happen per message. They
  are skipped for blocklisted senders, for low-info texts and while text scoring is off. The bio lookup never
  holds up a decision: it gives up after 1 second, and it is not sent at all while Telegram has `GetFullUser`
  flood-limited (Telethon would otherwise sleep up to 60 s inside the call). The message is then scored without
  the bio, and that sender's bio is retried after a minute. Telegram has no follower counts or account age, so those inputs
  stay "unknown", as they were for Telegram rows in training. Telegram training rows carry no bio, because the
  ingest does not fetch bios.
- **Action.** The action depends on whether the chat is private (a DM) or a group/channel (see below).
- **Vault.** Every nullified message is written to the vault first, then the action runs.
- **Logs.** Terminal logs show chat id, message id, score, reason, action and latency. They never show message
  text.

The sink never crashes on a single message. Errors are logged and the loop continues. Telegram rate limits
(`FloodWaitError`) are handled by sleeping for the time Telegram asks for and retrying. If the connection drops,
the sink reconnects with backoff (5 s doubling to 5 min; it resets only after a connection has stayed up for
2 minutes). If the session itself is dead (terminated in **Settings -> Devices**, logged out, or
`AUTH_KEY_DUPLICATED`), the sink exits with code 2 and tells you to run `make telegram-login`, instead of
retrying forever.

At start-up the sink logs the Telegram account id and username it is authorized as. Check it: when
`TG_SESSION` is set, the sink uses that StringSession, not the `data/telegram.session` that `make telegram-login`
wrote, so it acts on whichever account `TG_SESSION` belongs to.

### Configuration (`.env` or environment)

| variable | values | default |
| --- | --- | --- |
| `GVON_TG_DM_ACTION` | `delete_for_me` \| `mute` \| `archive` \| `log` | `delete_for_me` |
| `GVON_TG_GROUP_ACTION` | `delete_for_me` \| `mute` \| `archive` \| `log` | `log` |
| `GVON_TG_GROUP_DELETE` | `1` = delete for everyone in groups (not broadcast channels) where you are an admin with delete rights | off |
| `GVON_TG_MARK_READ` | `1` = after a fallback mute in a group, mark the chat read, only when the flagged message is its only unread message and never during `--backfill` (sends read receipts) | off |
| `GVON_TG_ALLOW` | comma-separated Telegram user ids or usernames the sink never acts on | empty |
| `GVON_TG_MEDIA_MAX_MB` | largest attachment downloaded to the vault before a delete; larger ones are logged instead of deleted | `50` |
| `GVON_DRY_RUN` | `1` = no Telegram calls at all (vault and logs only); same as `--dry-run` | off |
| `GVON_THRESHOLD`, `GVON_MODEL_DIR`, `GVON_BLOCKLIST`, `GVON_WATCH_DELTA` | as for the X proxy | |
| `GVON_TG_SESSION` | path of the session file | `data/telegram.session` |
| `TG_SESSION` | a Telethon StringSession used instead of the session file by `pull` AND the sink (never by `login`): the sink then acts on that account | unset |

An unknown action value stops the sink at start-up with an error, so a typo cannot fail silently.

### What each action does

| action | Telegram call | effect for you | effect for others |
| --- | --- | --- | --- |
| `delete_for_me` | `delete_messages(chat, [id], revoke=False)` only; the chat is not marked read | the message disappears from your chat history on all your devices; other unread messages stay unread | none: the sender and other members still have it, and no read receipt is sent |
| `mute` | notify settings `mute_until` = far future (muted "forever") for that chat | no more notifications from that chat; the message stays visible | none |
| `archive` | `mute`, then move the chat to the Archive folder (folder 1) | the chat leaves your main list and stays out of it | none |
| `log` | none | none: vault entry only | none |
| delete for everyone | `delete_messages(chat, [id], revoke=True)`; only with `GVON_TG_GROUP_DELETE=1` in a group where you are an admin with delete rights | the message is gone | **the message is gone for every member** |

Marking a chat read is never part of `delete_for_me`. Telegram's read acknowledgement marks every earlier message
in the chat read too and, in DMs and small groups, shows the senders their messages as read. With
`GVON_TG_MARK_READ=1` the sink marks read only after a fallback mute (where the message stays visible), only when
the flagged message is the chat's only unread message, and never during `--backfill`.

`archive` always mutes first, because Telegram moves an unmuted archived chat back to the main list when a new
message arrives. Each chat is muted and archived at most once per run, so a busy chat does not trigger an API call
per message. To undo `mute` or `archive`, unmute or unarchive the chat in any Telegram app.

## Limits

- **"Delete for me" only works in private chats and basic groups.** In supergroups and channels (most public groups
  and every group that has ever had more than 200 members or a public link), Telegram has no per-user deletion. A
  message there can only be deleted for everyone, which regular members cannot do for other people's messages.
  When `delete_for_me` is configured for such a chat, the sink falls back to muting the chat (result
  `fallback_mute (...)`). The message stays visible in that chat. In a basic group the same fallback applies when
  Telegram refuses the delete with a permission error. In a DM, and for any other error (a Telegram server error
  or timeout, for example), the chat's settings are left alone and the vault records `error: <name>`: one failed
  delete never mutes a whole conversation. **Muting and archiving are the only tools a non-admin has in
  supergroups.**
- **Delete for everyone needs admin rights.** `GVON_TG_GROUP_DELETE=1` only acts in groups (basic groups and
  supergroups) where the sink's admin check (cached for 10 minutes per chat) finds you are an admin, or the
  creator, with the "delete messages" right. It never acts in broadcast channels, even ones you admin, and never on
  a post sent as a channel: channel posts auto-forwarded into a discussion group (deleting one breaks its comment
  thread), or messages from a channel or anonymous admin posting as the group. Everywhere else the group action
  applies. This deletes the message for every member. Use it only in groups you moderate and where that is your
  call to make.
- **Unresolved chats are left alone.** If Telethon cannot resolve the chat of an incoming message, the sink
  falls back to its input peer; if that is missing too, the message is vaulted with result
  `skipped: unresolved chat` and nothing is done in Telegram.
- **Your other apps receive the message before the sink acts.** Telegram delivers the message to every logged-in
  session at once. Your phone may raise a push notification showing the text before the sink deletes it,
  typically within a few hundred ms, and a chat you already have open may render it for an instant. To keep DM
  text off your lock screen:
  - turn off message previews, or private-chat notifications entirely, in Telegram **Settings -> Notifications and
    Sounds** (Private Chats -> Show Notifications / Message Preview off), and rely on opening the app; or
  - use `GVON_TG_DM_ACTION=mute` or `archive` for chats you never want to hear from: once the sink has muted a
    chat, Telegram stops notifying for it on all devices.

  Notifications for groups usually are muted already. Leave them that way.
- **The sink must be running.** Nothing is filtered while the process is stopped or your machine is asleep.
  Messages that arrive in the meantime are not handled when it restarts unless you pass `--backfill N`, which scores
  the last N unread messages of every chat that has unread messages at start-up.
- **Only text is scored.** Captions count as text. Images, voice notes, stickers and video without a caption are
  not judged by the model, though their sender can still be on the blocklist. Before a captioned photo or
  document is deleted, the sink downloads it into the vault (see below); if it cannot, the message is only
  logged, never deleted.
- **The sink only matches Telegram accounts on the blocklist.** With the trained classifier loaded, the sink passes
  `platform="telegram"`, so a sender is blocked by id or @username only when a Telegram verdict put them on
  `data/blocklist.json` (`make telegram-ingest`, then `make label`). X accounts on the blocklist are not matched
  against Telegram senders, so until Telegram has been labelled the sink relies on the text model alone. In
  degraded mode (no model, `BlocklistDecider`) the platform is not checked: there a Telegram sender whose numeric
  id or username equals a blocked X account's is nullified even though they may be different people. Check the
  vault's `reason` field for `blocklist` hits.
- **Bots, channels and service messages.** Posts in broadcast channels you follow are incoming messages too. They
  get the group action (default `log`), never delete-for-everyone. Delete-for-me is impossible there. Use
  `mute`/`archive` or leave the channel. Telegram's own service messages (account `777000`) are never touched.
- **Edits are not re-scored.** A message that is later edited into something nasty is not seen again.

## The vault

Every nullified message is appended to `data/telegram_nullified.jsonl` **before** any action runs, so a crash
mid-action loses nothing. The file is created with mode `0600`. Each line holds:

```
ts, msg_date, chat_id, chat_title, is_private, msg_id, sender_id, sender_username, text,
entities (formatting / links / mentions, or null), reply_to_msg_id, fwd_from (forward header, or null),
media (null, or {type, path, mime_type, size, name, file_id} for photos/documents, {type, detail} for
       other media such as polls, contacts or locations, {type: "webpage"} for link previews),
score (null when not computed), reason ("blocklist" | "score" | "watchlist" | "author_aggregate"),
action, dry_run, backfill, result
```

**Attachments.** Before a delete (`delete_for_me` in a DM or basic group, or delete for everyone), a photo or
document attached to the message is downloaded to `data/telegram_media/<chat_id>/<msg_id>/` (directories `0700`,
file `0600`) and `media.path` records where. If the download fails, times out (2 min) or the file is larger than
`GVON_TG_MEDIA_MAX_MB` (default 50), the message is NOT deleted: it is logged with result
`logged (media not saved: <why>)`. Mute, archive and log keep the message in Telegram, so nothing is downloaded
for them. Stickers, polls and other media without a file are kept as metadata only.

`result` is what actually happened: `deleted_for_me`, `deleted_for_everyone`, `muted`, `archived`, `logged`,
`dry_run`, `fallback_mute (...)` (with ` + marked_read` when `GVON_TG_MARK_READ=1` marked the chat read),
`logged (media not saved: ...)`, `skipped: unresolved chat` or `error: <ExceptionName>`. For actions that call Telegram, the entry is written
with `result: "pending"` and a small `{"update": "result", ...}` line follows once the call returns. The `--vault`
reader folds the two together.

Audit false positives regularly:

```sh
make telegram-vault N=50
# or
.venv/bin/python -m gvon.telegram_nullifier --vault 50
```

This prints the last N entries, including the full text and the path of any saved attachment, to your terminal.
It does not connect to Telegram. A message deleted "for me" cannot be restored into the chat, but its text,
formatting, attachment, sender and chat stay in the vault. If a
sender keeps being nullified wrongly, raise `GVON_THRESHOLD`, retrain, or fix their verdict in the labels and
rebuild the blocklist (`make label LABEL_ARGS=--rebuild-only`).

The vault is the very content you asked not to see. Open it on purpose, not by accident.

## Privacy

- **`data/telegram.session` is a full login to your Telegram account.** Anyone who copies it can read all your
  chats and send messages as you, without your phone or 2FA password, until the session is terminated. GVON
  creates `data/` with mode `0700` and the session file with `0600`. `data/` is gitignored. Never commit it, sync it
  to a shared drive, paste it anywhere or include it in a backup you do not control. If it may have leaked,
  terminate the session in **Settings -> Devices** and run `make telegram-login` again.
- **The vault and the ingest pull hold real messages.** `data/telegram_nullified.jsonl`, `data/telegram_media/`
  and the Telegram ingest files contain other people's messages, attachments and ids. They stay under `data/` and never go into `fixtures/`, tests,
  issues or commits. Test data is synthetic.
- **Nothing leaves your machine except Telegram's own API calls.** The sink scores locally (CPU or Apple Metal) and
  talks only to Telegram's servers, as any Telegram client does. Labelling, if you run it on Telegram data, sends
  that text to the teacher model you configured, as for X.
- **Your account, your terms.** The sink is a user-session client ("userbot") running as you. Telegram's terms allow
  third-party clients built on the API. Run it on your own account only, at the normal pace it already keeps
  (it backs off on every FloodWait). Do not point it at other people's accounts.
