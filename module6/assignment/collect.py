#!/usr/bin/env python3
"""
Collect posts from the Bluesky Jetstream firehose.

Writes plain text to a file (one post per line).
"""

import argparse
import json
import sys

import websocket

FIREHOSE_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)
DEFAULT_NUM_POSTS = 2000
WEBSOCKET_TIMEOUT = 30


def collect_firehose(num_posts: int) -> list[bytes]:
    """Collect UTF-8 encoded posts from the Bluesky firehose."""
    posts: list[bytes] = []
    print(f"Connecting to firehose (target: {num_posts})...")
    ws = websocket.create_connection(FIREHOSE_URL, timeout=WEBSOCKET_TIMEOUT)
    try:
        while len(posts) < num_posts:
            msg = json.loads(ws.recv())
            if msg.get("kind") != "commit":
                continue
            text = msg.get("commit", {}).get("record", {}).get("text", "")
            if text:
                posts.append(text.encode("utf-8"))
                if len(posts) % 500 == 0:
                    print(f"  {len(posts):>6} / {num_posts}")
    finally:
        ws.close()
    print(f"  collected {len(posts)} posts")
    return posts


def write_file(posts: list[bytes], path: str) -> None:
    """Write posts as plain text, one per line."""
    with open(path, "w", encoding="utf-8") as f:
        for p in posts:
            # Replace newlines within posts with spaces to keep one-post-per-line
            line = p.decode("utf-8").replace("\n", " ").replace("\r", " ")
            f.write(line + "\n")
    print(f"  wrote {len(posts)} posts -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Bluesky posts as plain text",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="write text data to FILE",
    )
    p.add_argument(
        "--num-posts",
        "-n",
        type=int,
        default=DEFAULT_NUM_POSTS,
        help=f"posts to collect (default: {DEFAULT_NUM_POSTS})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    posts = collect_firehose(args.num_posts)
    write_file(posts, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
