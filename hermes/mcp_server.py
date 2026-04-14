#!/usr/bin/env python3
"""
EML Symbolic Regression -- MCP stdio server.

A lightweight Model Context Protocol server that exposes eml_symbolic_regression
as a tool over stdin/stdout. Configure in ~/.hermes/config.yaml:

    mcp_servers:
      eml:
        command: python
        args: ["/path/to/eml-symbolic-regression/hermes/mcp_server.py"]
        timeout: 120

Then Hermes auto-discovers the tool as mcp_eml_eml_symbolic_regression.

Protocol: JSON-RPC 2.0 over stdin/stdout (MCP stdio transport).
"""

import json
import sys
import numpy as np

TOOL_NAME = "eml_symbolic_regression"
TOOL_DESCRIPTION = (
    "Discover the mathematical formula underlying numerical data. "
    "Given x and y arrays, finds the simplest closed-form expression y=f(x) "
    "using EML binary trees. Returns the formula in standard math notation."
)
TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Input x values (at least 20 points, avoid x=0)",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Output y values corresponding to each x",
        },
        "max_depth": {
            "type": "integer",
            "default": 3,
            "description": "Max tree depth 1-4.",
        },
    },
    "required": ["x", "y"],
}


def handle_call(args):
    """Execute regression and return result dict."""
    from eml import regress

    x = np.array(args["x"], dtype=np.float64)
    y = np.array(args["y"], dtype=np.float64)
    max_depth = args.get("max_depth", 3)

    result = regress(x, y, max_depth=int(max_depth), verbose=False)
    mse = result["mse"]

    return {
        "expression": result["expression"],
        "eml_expression": result["eml_expression"],
        "depth": result["depth"],
        "mse": mse,
        "quality": "exact" if mse < 1e-8 else "approximate" if mse < 1e-1 else "rough",
        "constants": result["constants"],
    }


def send(msg):
    """Write a JSON-RPC message to stdout."""
    raw = json.dumps(msg)
    # MCP uses Content-Length framing
    sys.stdout.write(f"Content-Length: {len(raw)}\r\n\r\n{raw}")
    sys.stdout.flush()


def read():
    """Read a JSON-RPC message from stdin."""
    # Read headers
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    length = int(headers.get("content-length", 0))
    if length == 0:
        return None
    body = sys.stdin.read(length)
    return json.loads(body)


def main():
    """MCP stdio server loop."""
    while True:
        msg = read()
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "eml-symbolic-regression",
                        "version": "0.1.0",
                    },
                },
            })

        elif method == "notifications/initialized":
            pass  # no response needed

        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [{
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "inputSchema": TOOL_SCHEMA,
                    }],
                },
            })

        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name", "")
            args = params.get("arguments", {})

            if name != TOOL_NAME:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"},
                })
                continue

            try:
                result = handle_call(args)
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    },
                })
            except Exception as e:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                })

        else:
            if msg_id is not None:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                })


if __name__ == "__main__":
    main()
