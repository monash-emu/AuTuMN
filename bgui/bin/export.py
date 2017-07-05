import click
import requests
import sys
import os.path

from hashlib import sha224

@click.command()
@click.argument('server')
@click.argument('savelocation')
@click.option('--username', default='test',
              help="Username for logging on to the server. Default: test")
@click.option('--password', default='test',
              help="Password for logging on to the server. Default: test")
@click.option('--overwrite', default=False, type=bool,
              help="Whether or not to overwrite local projects with server ones. Default: False.")
def main(server, username, password, overwrite, savelocation):
    """
    A utility for downloading projects from Optima 2.0+ servers, per user.

    An example:

    \b
         python export.py --username=batman --password=batcar! http://athena.optimamodel.com batprojects

    The command above will log into http://athena.optimamodel.com as the user
    'batman' with the password 'batcar!', and download all of that user's
    projects into the folder 'batprojects' in the current directory.
    """
    # Make sure that we don't send duplicate /s
    if server[-1:] == "/":
        server = server[:-1]

    old_session = requests.Session()

    click.echo('Logging in as %s...' % (username,))
    hashed_password = sha224()
    hashed_password.update(password)
    password = hashed_password.hexdigest()

    # Old server login
    old_login = old_session.post(server + "/api/user/login",
                                 json={'username': username,
                                       'password': password})
    if not old_login.status_code == 200:
        click.echo("Failed login:\n%s" % (old_login.content,))
        sys.exit(1)
    click.echo("Logged in as %s on old server" % (old_login.json()["displayName"],))

    old_projects = old_session.get(server + "/api/project").json()["projects"]
    click.echo("Downloading projects...")

    if not savelocation:
        project_path = '%sprojects' % (username,)
    else:
        project_path = savelocation

    try:
        os.makedirs(project_path)
    except:
        pass

    for project in old_projects:
        click.echo("Downloading project '%s'" % (project["name"],))
        url = server + "/api/project/" + project["id"] + "/data"

        if os.path.isfile(project_path +  "/" + project["name"] + ".prj"):
            if overwrite:
                click.echo("Downloaded already, overwriting...")
            else:
                click.echo("Downloaded already, skipping (set --overwrite=True if you want to overwrite)")
                continue

        download = old_session.get(url)

        with open(project_path + "/" + project["name"] + ".prj", 'wb') as f:
            f.write(download.content)


    click.echo("All downloaded! In folder %s" % (project_path,))

if __name__ == '__main__':
    main()
