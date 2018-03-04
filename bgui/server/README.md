
# AuTUMN web-server

The web-server is a Python Flask app, which is synchronous, that is run
through a Twisted Python asynchronous server. The Flask app uses SQLAlchemy
to access the database, which is by default an SQLite file. 
Uploaded files are saved in the SAVE_FOLDER config option.

The compiled web-client is found in the `../client/dist` directory. To 
develop the web-client, go to `../client`.

## How to run

To install:

    > pip install -r requirements.txt
    
To run the server:

    > python run_server.py
    
Then open the webserver:

    http://localhost:3000
    
## Configuring the web-server

To setup the SQLAlchemy database, and the location where files are 
uplaoded, edit `config.py`:

```
SQLALCHEMY_DATABASE_URI = 'sqlite:///database.sqlite'
SECRET_KEY = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'
SAVE_FOLDER = '/tmp/autumn'
```

The config `config.py` is loaded in `conn.py`.

The port is set in `_twisted_wsgi.py`. Make sure that this port 
corresponds to the port expected in the web-client.

## Files

- `run_server.py` - entry point to web-server, starts up Twisted
- `api.py` - definition of the Flask app 
- `config.py` - configuration of the SQLAlchemy database and save folder
- `handlers.py` - the URL handlers for the RPC-JSON protocol
- `conn.py` -  central place to store Flask and SQLAlchemy global variables
- `dbmodel.py` - SQLALchemy database definition and accessor function
