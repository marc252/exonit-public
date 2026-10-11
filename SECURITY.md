# Security policy

ExoNit is an experimental research system intended for a trusted private network. It is not hardened for
exposure to the internet.

## Reporting a vulnerability

Please report security problems privately to **<security contact email: to be filled in by the maintainer>**.
Do not open a public issue for a vulnerability. Include the affected component, steps to reproduce, and the
impact you see. Expect an acknowledgement within a few days; this is a small project, so there is no formal
response SLA.

## Operating it safely

- **Change the worker token.** `EXONIT_WORKER_TOKEN` is the shared secret between the server and every worker, and
  it also protects the admin endpoints. The built-in default is a placeholder (`exonit-default-token`) that is
  published in this repository: never run a server that still accepts it outside a throwaway test. Generate a long
  random value, put it in `.env` (mode 0600), in each worker's `config.json`, and in the dashboard's browser
  storage.
- **Do not expose port 8400 to the internet.** Keep the server on a private network or behind a VPN or an
  authenticating reverse proxy with TLS. The API speaks plain HTTP and bearer tokens.
- **Keep PostgreSQL and Redis private.** Bind them to localhost or a private interface and use a non-default
  password for the `exonit` role.
- **Protect secrets.** `.env`, `/etc/default/exonit-compactd`, S3 keys and `ANTHROPIC_API_KEY` must never be
  committed. The repository's `.gitignore` excludes `.env`; the release process also scans for secrets.
- **SSH.** `farmctl` uses SSH public-key authentication to reach the workers. Use dedicated keys and restrict them
  to the worker accounts.
- **Workers run server-supplied code.** Auto-update downloads and executes the `worker/` tree served by the server.
  Treat the server as the trust root of your fleet, and only connect workers you control to a server you control.
- **Autonomy supervisor.** It can start and stop the fleet. It is shadow-mode by default and needs two explicit
  keys to act; leave it in shadow mode unless you have read the campaign guide.
- **Docker** (optional ExoMiner path) pulls an image from a public registry; pin it to a digest.

## Supported versions

Only the current `main` branch is supported.
