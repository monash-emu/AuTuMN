"""
Create a static site from our S3 bucket.
"""
import os
import io
from typing import List
from datetime import datetime

import boto3
import timeago

client = boto3.client("s3")

BUCKET = "autumn-calibrations"
WEBSITE_URL = "http://autumn-calibrations.s3-website-ap-southeast-2.amazonaws.com"
BUCKET_URL = "https://autumn-calibrations.s3-ap-southeast-2.amazonaws.com/"


def get_url(path: str):
    if path.startswith("http"):
        return path
    else:
        return f"{WEBSITE_URL}/{path}"


def get_pretty_name(s: str):
    """
    Get a pretty run name from the slug stored in AWS S3
    """
    if "-" in s:
        model_name, timestamp, branch, commit = s.split("-")
        run_dt = datetime.fromtimestamp(int(timestamp))
        run_dt_str = run_dt.strftime("%a at %H:%M %d-%m-%Y")
        model_name = model_name.title()
        return f"{model_name} {run_dt_str} ({branch} {commit[:7]})"
    else:
        return s.title()


def update_website():
    response = client.list_objects_v2(Bucket=BUCKET)
    objects = [o for o in response["Contents"] if not o["Key"].endswith(".html")]
    keys = [o["Key"] for o in objects]
    runs = {k.split("/")[0] for k in keys}
    models = sorted(list({r.split("-")[0] for r in runs}))

    # Render the main page
    html = ""
    for model in models:
        model_html = ""
        model_name = model.title()
        model_html += render_header(f"{model_name} runs")
        model_runs = reversed(sorted([r for r in runs if model in r]))
        link_htmls = []
        for r in model_runs:
            run_name = get_pretty_name(r)
            run_url = get_url(r)
            link_html = render_link(run_name, run_url)
            link_htmls.append(link_html)

        model_html += render_list(link_htmls)
        html += model_html

    html = render_body(html)
    upload_html(html, "index.html")

    for run in runs:
        children = ["/".join(k.split("/")[1:]) for k in keys if k.startswith(run)]
        update_page(run, children)


def update_page(path: str, children: List[str]):
    print("Updating page", path)
    html = ""
    title = get_pretty_name(path.split("/")[-1])
    html += render_header(title)
    dirs = set()
    files = []
    for child in children:
        parts = child.split("/")
        if len(parts) > 1:
            dirs.add(parts[0])
        else:
            files.append(child)

    link_htmls = []
    dirs = sorted(list(dirs))
    for dirname in dirs:
        dirpath = os.path.join(path, dirname)
        link_html = render_link(dirname, dirpath)
        link_htmls.append(link_html)

    if files:
        download_html = render_download_all_link(path, files)
        link_htmls.append(download_html)

    for file in files:
        filepath = os.path.join(path, file)
        link_html = render_link(file, filepath)
        link_htmls.append(link_html)

    html += render_list(link_htmls)
    html = render_body(html)
    key = os.path.join(path, "index.html")
    upload_html(html, key)

    for dirname in dirs:
        dir_children = [
            "/".join(k.split("/")[1:]) for k in children if k.startswith(dirname)
        ]
        dirpath = os.path.join(path, dirname)
        update_page(dirpath, dir_children)


def upload_html(s: str, key: str):
    f = io.BytesIO(s.encode("utf-8"))
    client.upload_fileobj(f, BUCKET, key, ExtraArgs={"ContentType": "text/html"})


def render_header(text: str):
    return f"<h1 class='ui header'>{text}</h1>"


def render_link(text: str, url: str):
    url = get_url(url)
    return f"<a href='{url}'>{text}</a>"


def render_list(items: List[str]):
    items_str = "\n".join(f"<li>{item}</li>" for item in items)
    return f"<ul>{items_str}</ul>"


def render_body(inner: str):
    # See https://semantic-ui.com/
    return f"""
    <html>
        <head>
            <title>Autumn Calibrations</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">
            <script src="https://code.jquery.com/jquery-3.1.1.min.js" crossorigin="anonymous"></script>
            <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.js"></script>
        </head>
        <body>
            <div class="ui inverted menu">
                <a class="item" href={WEBSITE_URL}>Autumn Calibrations</a>
            </div>
            <div class="ui container">
                {inner}
            </div>
        </body>
    </html>
    """


def render_download_all_link(path, files):
    urls = [get_url(os.path.join(path, file)) for file in files]
    urls_str = ",".join([f"'{url}'" for url in urls])
    return f"""
    <a href="#" id="downloadall">Download all files</a>
    <script>
    let urls = [{urls_str}]
    $('#downloadall').click(function(e) {{
        e.preventDefault();
        let link = document.createElement('a');
        link.style.display = 'none';
        document.body.appendChild(link);
        for (let i = 0; i < urls.length; i++) {{
            link.setAttribute('download', urls[i].split('/').pop());
            link.setAttribute('href', urls[i]);
            link.click();
        }}
        document.body.removeChild(link);
    }});
    </script>
    """
