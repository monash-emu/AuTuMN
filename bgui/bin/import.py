import click
import requests
import sys
import os.path
import glob

from hashlib import sha224

@click.command()
@click.argument('server')
@click.argument('project_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--username', default='test',
              help="Username for logging on to the server. Default: test")
@click.option('--password', default='test',
              help="Password for logging on to the server. Default: test")
@click.option('--overwrite', default=False, type=bool,
              help="Whether or not to overwrite server projects with local ones. Default: False")
def main(server, project_paths, username, password, overwrite):
    """
    A utility for mass-uploading projects to Optima 2.0+ servers, per user.

    An example:

    \b
        python import.py --username=robin --password=robinerang http://newdb.optimamodel.com batprojects/

    The command above will log into http://newdb.optimamodel.com as the user
    'robin' with the password 'robinerang' and upload all the projects inside
    the batprojects directory. Individual projects can be given, and multiple
    can be passed one after the other.

    The --overwrite flag will replace server projects named the same as local
    .prj files with the local project.
    """
    # Make sure that we don't send duplicate /s
    if server[-1:] == "/":
        server = server[:-1]

    # Make sure we've got a list of files, not folders
    project_paths_tmp = list(set([click.format_filename(x) for x in project_paths]))
    project_paths = []

    for x in project_paths_tmp:
        if not os.path.isfile(x):
            project_paths.extend(glob.glob(x + "/*.prj"))
        else:
            project_paths.append(x)

    click.echo("Preparing to upload %s projects to %s..." % (len(project_paths), server))

    new_session = requests.Session()

    click.echo('Logging in as %s...' % (username,))
    hashed_password = sha224()
    hashed_password.update(password)
    password = hashed_password.hexdigest()

    # New server login
    new_login = new_session.post(server + "/api/user/login",
                                 json={'username': username,
                                       'password': password})
    if not new_login.status_code == 200:
        click.echo("Failed login:\n%s" % (new_login.content,))
        sys.exit(1)
    click.echo("Logged in as %s on new server" % (new_login.json()["displayName"],))

    click.echo("Uploading...")
    click.echo("First, getting the projects off the new server.")

    new_projects = new_session.get(server + "/api/project").json()["projects"]
    projects = {x["name"]:x for x in new_projects}

    for project_path in project_paths:
        project_name = os.path.basename(project_path)[:-4]

        f = open(project_path, 'rb')

        if project_name in projects.keys():
            if overwrite:
                project_upload = new_session.post(
                    server + "/api/project/" + projects[project_name]["id"] + "/data",
                    files={"file": (project_name + ".prj", f)})
                click.echo("Uploaded + overwrote %s" % (project_name,))
                assert project_upload.status_code == 200, project_upload.status_code
            else:
                click.echo("NOT UPLOADING %s because --overwrite=False" % (project_name,))
                f.close()
                continue

        else:
            # New upload
            new_project_upload = new_session.post(
                server + "/api/project/data",
                data={"name": project_name},
                files={"file": (project_name + ".prj", f)})
            assert new_project_upload.status_code == 200, new_project_upload.status_code
            click.echo("Uploaded %s" % (project_name,))

        f.close()


if __name__ == '__main__':
    main()
