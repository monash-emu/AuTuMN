"""
Create a static site from our S3 bucket.
"""
import boto3
import timeago

client = boto3.client("s3")

BUCKET = "autumn-calibrations"


def update_website():
    response = client.list_objects_v2(Bucket=BUCKET,)
    # TODO
    """
    {'Contents': [{'ETag': '"b8c5dacac31146f48b078758a08cf83c"',
                'Key': 'malaysia-1591072443-master-b34ccc0cb669db6a9661f9bba4301a4b922104ba/data/calibration_outputs/outputs_calibration_chain_1.db',
                'LastModified': datetime.datetime(2020, 6, 2, 4, 35, 42, tzinfo=tzutc()),
                'Size': 4198400,
                'StorageClass': 'STANDARD'},
                {'ETag': '"27ede07f947ec1ee5a363ca9a47a5cb2"',
                'Key': 'malaysia-1591072443-master-b34ccc0cb669db6a9661f9bba4301a4b922104ba/data/calibration_outputs/outputs_calibration_chain_2.db',
                'LastModified': datetime.datetime(2020, 6, 2, 4, 35, 42, tzinfo=tzutc()),
                'Size': 2830336,
                'StorageClass': 'STANDARD'},

                {'ETag': '"bdb50fef556e03c6d51db722e365cb72"',
                'Key': 'philippines-1591094593-master-3d21352cec14a249b6bf9b17e8471877dbb1c6d5/logs/run-9.log',
                'LastModified': datetime.datetime(2020, 6, 2, 13, 47, 27, tzinfo=tzutc()),
                'Size': 117920,
                'StorageClass': 'STANDARD'}],
    'EncodingType': 'url',
    'IsTruncated': False,
    'KeyCount': 269,
    'MaxKeys': 1000,
    'Name': 'autumn-calibrations',
    'Prefix': '',
    'ResponseMetadata': {'HTTPHeaders': {'content-type': 'application/xml',
                                        'date': 'Tue, 02 Jun 2020 23:24:00 GMT',
                                        'server': 'AmazonS3',
                                        'transfer-encoding': 'chunked',
                                        'x-amz-bucket-region': 'ap-southeast-2',
                                        'x-amz-id-2': 'LSEza0ndUpxzZeDxZHJ5zBkce+O4LNv2felCLuM8xLYqEeLniH0qKaW6iACZUJOKrK1wDtenp9Q=',
                                        'x-amz-request-id': 'F28FF083F70183B1'},
                        'HTTPStatusCode': 200,
                        'HostId': 'LSEza0ndUpxzZeDxZHJ5zBkce+O4LNv2felCLuM8xLYqEeLniH0qKaW6iACZUJOKrK1wDtenp9Q=',
                        'RequestId': 'F28FF083F70183B1',
                        'RetryAttempts': 0}}
    """


# <a href="#" class="yourlink">Download</a>
# $('a.yourlink').click(function(e) {
#     e.preventDefault();
#     window.open('mysite.com/file1');
#     window.open('mysite.com/file2');
#     window.open('mysite.com/file3');
# });
