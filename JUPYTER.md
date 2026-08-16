# Jupyter setup (numerai root)

Standardized so any future session (Claude or human) can connect without hunting for a
token. This is the single server for the whole `numerai/` folder: datasets live here
once (`datasets/`), and each research effort gets its own subfolder (e.g.
`feature-neutralization/`) with its own notebooks, all reachable from one server
rooted at `numerai/`.

`feature-neutralization/` used to run its own dedicated server and `.mcp.json` on the
same port; those are retired in favor of this one. Its notebooks are still reachable
as relative paths from this server's root, e.g.
`feature-neutralization/code/feature neutralization.ipynb`. A Claude Code session
working in `feature-neutralization/` should be started from the `numerai/` root (or
otherwise pick up this root `.mcp.json`) so it can find the `jupyter` MCP server —
there's no more per-subfolder `.mcp.json` to fall back on.

## Quick start (what to do each time you sit down to work)

1. `cd` into `numerai/` (or open your Claude Code session rooted there — see the
   `feature-neutralization/` note above if you work from a subfolder).
2. Run `./start_jupyter.sh`. It's idempotent, so it's safe to run even if the server
   is already up — it'll just print that and exit.
3. Open a Claude Code session in `numerai/` (or a subfolder that inherits its
   `.mcp.json`). The `jupyter` MCP server starts automatically from `.mcp.json` — no
   manual `connect_to_jupyter` call needed.
4. In the session, attach to the notebook you're working on with `use_notebook`, e.g.
   `use_notebook(document_id="feature-neutralization/code/feature neutralization.ipynb")`
   or a path under a new research subfolder.
5. If you also want to work in the browser, JupyterLab itself is at
   `http://127.0.0.1:8888/lab` (no token needed).

That's it — steps 2-3 are what "launch Jupyter + MCP" means day to day; the rest of
this doc is reference detail.

## The standard

- **Port:** always `8888`, bound to `127.0.0.1` only (not reachable off-machine).
- **Auth:** disabled (`--ServerApp.token=''`). Safe because it's loopback-only.
- **Interpreter:** `~/ml-venv` (has numpy/pandas/sklearn etc. already installed).
  Launching Jupyter itself from `~/ml-venv/bin/jupyter` means the default kernel
  already uses that venv.
- **Server root:** the `numerai/` folder itself, so every subfolder (`datasets/`,
  `feature-neutralization/`, future research folders, ...) is reachable as a relative
  path from the Lab file browser.

## Starting it

```
./start_jupyter.sh
```

Idempotent — checks if something's already listening on 8888 and exits quietly if so.
Logs to `jupyter.log`.

## Connecting an MCP session

`.mcp.json` at the numerai root already points the `jupyter` MCP server at
`http://127.0.0.1:8888` with an empty token, so a fresh Claude Code session started
from this folder should just work once the server is running — no manual
`connect_to_jupyter` call needed. If it wasn't started yet, run `./start_jupyter.sh`
first (or have the agent run it).

No `DOCUMENT_ID` is pinned in the root `.mcp.json`, since which notebook you're working
on changes per research folder. Use the MCP `use_notebook` tool (or list/select
interactively) to attach to whichever notebook you're using in the current session,
e.g. `use_notebook(document_id="feature-neutralization/code/feature neutralization.ipynb")`.

If connecting manually (e.g. a different MCP client), the call is:

```
connect_to_jupyter(jupyter_url="http://127.0.0.1:8888")
```

(no token argument needed).

## Gotchas learned the hard way

- Launching Jupyter without checking for an existing instance on the same port causes
  it to silently fall back to the next free port (8889, 8890, ...) instead of erroring.
  Killing stray servers/kernels and always going through `start_jupyter.sh` avoids this.
- Restarting the Jupyter server invalidates any kernel a previously-connected MCP
  session was attached to. If `execute_code` goes silent / `list_kernels` shows none,
  call `unuse_notebook` then `use_notebook` again to get a fresh kernel.
- Port 8888 is a common default for other unrelated Jupyter projects on this machine
  too (not just ones under `numerai/`). If `start_jupyter.sh` reports "already running"
  unexpectedly, check `ps aux | grep jupyter-lab` for the `--ServerApp.root_dir` before
  assuming it's this server.
