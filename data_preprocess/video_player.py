from flask import Flask, request, Response, render_template_string, abort
import os
import mimetypes
import argparse

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Local Video Player</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0f1115;
      --panel: #161922;
      --border: #2a2f3a;
      --text: #e6e8ee;
      --muted: #9aa1b2;
      --accent: #4c8dff;
      --error: #ff5d5d;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont,
                   "Segoe UI", Roboto, "PingFang SC",
                   "Microsoft YaHei", sans-serif;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 600;
    }

    .subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 18px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }

    .path-row {
      display: flex;
      gap: 10px;
    }

    input {
      flex: 1;
      background: #0c0f14;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }

    input::placeholder {
      color: #6b7280;
    }

    input:focus {
      border-color: var(--accent);
    }

    button {
      background: var(--accent);
      border: none;
      color: white;
      padding: 0 18px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
      white-space: nowrap;
    }

    button:hover {
      filter: brightness(1.05);
    }

    .hint {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }

    .error {
      margin-top: 8px;
      font-size: 13px;
      color: var(--error);
      display: none;
    }

    .video-wrap {
      margin-top: 18px;
      background: black;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    video {
      width: 100%;
      max-height: 70vh;
      display: block;
    }

    footer {
      margin-top: 20px;
      font-size: 12px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Local Video Player</h1>
    <div class="subtitle">
      输入本地视频的绝对路径（通过 Flask 本地服务播放）
    </div>

    <div class="panel">
      <div class="path-row">
        <input id="path"
               placeholder="C:\\movies\\test.mp4  或  /home/user/video.mp4">
        <button onclick="play()">播放</button>
      </div>
      <div class="hint">
        支持拖动进度条 · 支持大文件 · 不经过 VSCode
      </div>
      <div class="error" id="error">
        文件不存在或无法读取
      </div>
    </div>

    <div class="video-wrap">
      <video id="video" controls></video>
    </div>

    <footer>
      Flask · HTML5 Video · Local Only
    </footer>
  </div>

  <script>
    const input = document.getElementById("path");
    const video = document.getElementById("video");
    const error = document.getElementById("error");

    function play() {
      const path = input.value.trim();
      if (!path) return;

      error.style.display = "none";
      video.src = "/video?path=" + encodeURIComponent(path);
      video.load();
      video.play().catch(() => {
        error.style.display = "block";
      });
    }

    input.addEventListener("keydown", e => {
      if (e.key === "Enter") play();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video")
def video():
    path = request.args.get("path")
    if not path or not os.path.isfile(path):
        abort(404)

    file_size = os.path.getsize(path)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "video/mp4"

    range_header = request.headers.get("Range")
    if range_header:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        length = end - start + 1

        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(data, 206, mimetype=mime)
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    else:
        with open(path, "rb") as f:
            data = f.read()
        resp = Response(data, 200, mimetype=mime)

    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(len(data))
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)
