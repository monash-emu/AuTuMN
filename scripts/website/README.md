# Autumn Results Website

This is a static website built to allow users to explore Autumn calibration results that are stored in AWS S3.
It is also hosted on AWS S3 [here](http://autumn-data.s3-website-ap-southeast-2.amazonaws.com/)

This site is built using [NextJS](https://nextjs.org/).

### Development

You will need NodeJS and Yarn or NPM package managers installed.

To get started, from this directory

```bash
# Pull data from AWS, requires an "autumn" AWS credentials profile.
./read_website_data.py

# Install JavaScript packages
yarn install

# Start local dev server on http://localhost:3000
yarn dev
```

### Deployment

```bash
./deploy.sh
```
