# Setup EC2 instance profile "worker-profile"
S3_FULL_ACCESS_ARN=arn:aws:iam::aws:policy/AmazonS3FullAccess
ROLE_NAME=worker-role
PROFILE_NAME=worker-profile
ROLE_DOC="{
  \"Version\": \"2012-10-17\",
  \"Statement\": {
    \"Effect\": \"Allow\",
    \"Principal\": {\"Service\": \"ec2.amazonaws.com\"},
    \"Action\": \"sts:AssumeRole\"
  }
}"
aws --profile autumn iam \
    create-role \
    --role-name $ROLE_NAME \
    --description "Autumn worker" \
    --assume-role-policy-document "$ROLE_DOC"

aws --profile autumn iam \
    attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn $S3_FULL_ACCESS_ARN

aws --profile autumn iam \
    create-instance-profile \
    --instance-profile-name $PROFILE_NAME

aws --profile autumn iam \
    add-role-to-instance-profile \
    --instance-profile-name $PROFILE_NAME \
    --role-name $ROLE_NAME 

aws --profile autumn iam \
    get-instance-profile \
    --instance-profile-name $PROFILE_NAME