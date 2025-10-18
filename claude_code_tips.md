# Tips for using Cruijff_kit with Claude Code and VS Code

## Tips for Claude and VS Code SSH connection stability

Problem: Sometiems you will have long running sessions that get disconnected.

Solution: This will send a "Stay Alive" message every [60] seconds, and do keep the connection active for [120] signals.

Be sure to replace your.server.address with the right address for example: della-vis1.princeton.edu
```
cat >> ~/.ssh/config << 'EOF'

Host your-server-name
  HostName your.server.address
  ServerAliveInterval 60
  ServerAliveCountMax 120
EOF
```

## To create a CLAUDE.md

run /init in Claude Code.

## To add mcp servers to cruijff_kit

## to add mcp servers for you using cruijff_kit
