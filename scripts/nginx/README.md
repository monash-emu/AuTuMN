### Autumn website proxy

The autumn website at www.autumn-data.com is served through an NGINX proxy so that we can restrict access to certain areas.

The NGINX proxy lives on the Buildkite AWS server which has a static IP of `13.55.240.146`

### Setup

Install NGINX

```bash
apt install nginx apache2-utils -y
```

### Update

```
scp -i ~/.ssh/autumn.pem autumn.conf root@13.55.240.146:/etc/nginx/sites-enabled/
ssh -i ~/.ssh/autumn.pem root@13.55.240.146 "nginx -s reload"
```

### Basic auth passwords

See docs on passwords [here](https://docs.nginx.com/nginx/admin-guide/security-controls/configuring-http-basic-authentication/)

Current users are:

- bicol in /etc/nginx/bicol.passwords

To create a password file

``` bash
htpasswd -c /etc/nginx/bicol.passwords bicol
```

To add a new user to the file

``` bash
htpasswd /etc/nginx/bicol.passwords user2
```
